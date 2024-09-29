import multiprocessing as mp
from random import randrange

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def identity(x):
    return x


class DistributedDalaloaderWrapper:
    def __init__(self, dataloader: DataLoader, collate_fn):
        self.dataloader = dataloader
        self.collate_fn = collate_fn

    def _epoch_iterator(self, it):
        for batch in it:
            yield self.collate_fn(batch)

    def __iter__(self):
        it = iter(self.dataloader)
        return self._epoch_iterator(it)

    def __len__(self):
        return len(self.dataloader)

    @property
    def dataset(self):
        return self.dataloader.dataset

    def set_epoch(self, epoch: int):
        self.dataloader.sampler.set_epoch(epoch)


def universal_dict_collater(batch):
    keys = batch[0].keys()
    all_data = {key: [] for key in keys}
    for one_batch in batch:
        for key in keys:
            all_data[key].append(one_batch[key])
    return all_data


def pad_collate_fn(batch):
    max_length = max(sample["waveform"].shape[-1] for sample in batch)

    for sample in batch:
        waveform = sample["waveform"]
        length = waveform.shape[-1]
        if length < max_length:
            padding_length = max_length - length
            waveform = F.pad(waveform, (0, padding_length), "constant", value=0.0)
        sample["waveform"] = waveform

        # Create padding mask
        padding_mask = torch.full((1, max_length), fill_value=False, dtype=torch.bool)
        if length < max_length:
            padding_mask[:, -padding_length:] = True
        sample["padding_mask"] = padding_mask.long()

    waveforms = torch.stack([sample["waveform"] for sample in batch]).squeeze(1)
    padding_masks = torch.stack([sample["padding_mask"] for sample in batch]).squeeze(1)
    emotions = torch.stack([sample["emotion"] for sample in batch]).squeeze(1)

    return {
        "waveform": waveforms,
        "padding_mask": padding_masks,
        "emotion": emotions,
    }


class DataloaderFactory:
    def __init__(self, args):
        self.args = args

    def build(self, size, state: str = "train", bs: int = 1):
        dataset = IemocapDataset(args=self.args, state=state, size=size)
        collate_fn = pad_collate_fn
        if self.args.distributed:
            sampler = DistributedSampler(dataset, shuffle=state == "train")
        else:
            sampler = None
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=bs,
            drop_last=False,
            num_workers=self.args.num_workers,
            collate_fn=identity,
            sampler=sampler,
            pin_memory=True,
            multiprocessing_context=mp.get_context(
                "fork"
            ),  # fork/spawn # quicker! Used with multi-process loading (num_workers > 0)
        )

        return DistributedDalaloaderWrapper(dataloader, collate_fn)


class DownstreamDataset(Dataset):
    def __init__(
        self, df, wavdir, col_sample="file_path", col_label="label", size=None
    ):
        self.df = df
        self.wavdir = wavdir
        self.col_sample = col_sample
        self.col_label = col_label
        self.size = size

    def __len__(self):
        return len(self.df) if self.size is None else self.size

    def __getitem__(self, idx):
        filename = self.df.loc[idx, self.col_sample]
        waveform = torch.load(f"{self.wavdir}{filename}")
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        emotion = torch.tensor(
            [int(self.df.loc[idx, self.col_label])], dtype=torch.long
        )

        sample = {
            "waveform": waveform,
            "emotion": emotion,
        }

        return sample


class IemocapDataset(DownstreamDataset):
    def __init__(self, size, args, state: str = "train"):
        _mapping = {
            "ang": 0,
            "neu": 1,
            "hap": 2,
            "sad": 3,
        }
        classes = args.classes[: args.nClasses]
        df = pd.read_csv(args.train_list)
        wavdir = args.train_path
        batch_length = args.batch_length if state == "train" else args.val_batch_length
        if _mapping is not None:
            df["emotion"] = df["emotion"].map(_mapping).astype(np.float32)
            df = df.loc[df["emotion"].notnull()]
            df = df.loc[df["emotion"].isin(classes)]
        if state == "train":
            df = df[~df["session"].isin([args.eval_session])]
        elif state == "val":
            df = df[df["session"] == args.eval_session]
            df = df[~df["gender"].isin([args.test_gender])]
        else:
            df = df[df["session"] == args.eval_session]
            df = df[df["gender"] == args.test_gender]
        df = df.reset_index()

        super().__init__(
            df,
            wavdir,
            col_sample="file_path",
            col_label="emotion",
            size=size,
        )

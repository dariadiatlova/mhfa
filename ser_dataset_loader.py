import multiprocessing as mp
from random import randrange

import numpy as np
import pandas as pd
import torch
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


class DataloaderFactory:
    def __init__(self, args):
        self.args = args

    def build(self, size, state: str = "train", bs: int = 1):
        dataset = IemocapDataset(args=self.args, state=state, size=size)
        collate_fn = universal_dict_collater
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
        self,
        df,
        wavdir,
        batch_length,
        col_sample="file_path",
        col_label="label",
        random_crop=True,
        size=None,
    ):
        self.df = df
        self.wavdir = wavdir
        self.batch_length = batch_length
        self.col_sample = col_sample
        self.col_label = col_label
        self.random_crop = random_crop
        self.size = size

    def __len__(self):
        return len(self.df) if self.size is None else self.size

    def _random_crop(self, waveform):
        max_start_index = waveform.shape[1] - self.batch_length
        try:
            random_start = randrange(max_start_index)
        except Exception as e:
            print(e)
            print(f"Shape: {waveform.shape[1]}, start index: {max_start_index}")
        cropped_waveform = waveform[:, random_start : random_start + self.batch_length]
        assert cropped_waveform.shape[1] == self.batch_length, (
            f"Expected cropped shape to be {self.batch_length}, "
            f"got: {cropped_waveform.shape[1]}, "
            f"start index: {random_start}"
        )
        return cropped_waveform

    def __getitem__(self, idx):
        filename = self.df.loc[idx, self.col_sample]  # .split("/")[-1]
        waveform = torch.load(f"{self.wavdir}{filename}")
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        emotion = torch.tensor(
            [int(self.df.loc[idx, self.col_label])], dtype=torch.long
        )
        padding_mask = torch.full(
            (1, self.batch_length), fill_value=False, dtype=torch.bool
        )

        length = waveform.shape[-1]
        if length > self.batch_length:
            if self.random_crop:
                waveform = self._random_crop(waveform)  # train
            else:
                waveform = waveform[:, : self.batch_length]  # validation
        elif length < self.batch_length:
            padding_length = self.batch_length - length
            waveform = torch.nn.functional.pad(
                waveform, (0, padding_length), "constant", value=0.0
            )
            padding_mask[:, -padding_length:] = True

        sample = {
            "waveform": waveform,
            "padding_mask": padding_mask,
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
            batch_length,
            col_sample="file_path",
            col_label="emotion",
            random_crop=args.random_crop if state == "train" else False,
            size=size,
        )

#! /usr/bin/python
# -*- encoding: utf-8 -*-

import glob
import math
import os
import pdb
import random
import threading
import time

import numpy as np
import pandas as pd
import soundfile
import torch
import torch.distributed as dist
import tqdm
# import soundfile
from scipy import signal
from torch.utils.data import DataLoader, Dataset


def round_down(num, divisor):
    return num - (num % divisor)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def loadWAV(filename, max_frames, evalmode=True, num_eval=10):

    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # # Read wav file and convert to torch tensor
    # audio, sample_rate = soundfile.read(filename)

    # Read pt tensor
    audio = torch.load(filename, weights_only=False).squeeze(0).numpy()

    audiosize = audio.shape[0]

    padding_mask = None

    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = np.pad(audio, (0, shortage), "wrap")
        audiosize = audio.shape[0]
        padding_mask = np.zeros(audiosize, dtype=np.float32)
        padding_mask[audiosize - shortage :] = 1  # Mark the padded regions

    if evalmode:
        startframe = np.linspace(0, audiosize - max_audio, num=num_eval)
    else:
        startframe = np.array([np.int64(random.random() * (audiosize - max_audio))])

    feats = []
    masks = []
    if evalmode and max_frames == 0:
        feats.append(audio)
        masks.append(padding_mask)
    else:
        for asf in startframe:
            feats.append(audio[int(asf) : int(asf) + max_audio])
            if padding_mask is not None:
                masks.append(padding_mask[int(asf) : int(asf) + max_audio])
            else:
                masks.append(np.zeros(max_audio, dtype=np.float32))

    feat = np.stack(feats, axis=0).astype(np.float32)
    mask = np.stack(masks, axis=0).astype(np.float32)

    return feat, mask


class AugmentWAV(object):

    def __init__(self, musan_path, rir_path, max_frames):

        self.max_frames = max_frames
        self.max_audio = max_audio = max_frames * 160 + 240

        self.noisetypes = ["noise", "speech", "music"]

        self.noisesnr = {"noise": [0, 15], "speech": [13, 20], "music": [5, 15]}
        self.numnoise = {"noise": [1, 1], "speech": [3, 8], "music": [1, 1]}
        self.noiselist = {}

        augment_files = glob.glob(os.path.join(musan_path, "*/*/*/*.wav"))

        for file in augment_files:
            if not file.split("/")[-4] in self.noiselist:
                self.noiselist[file.split("/")[-4]] = []
            self.noiselist[file.split("/")[-4]].append(file)

        self.rir_files = glob.glob(os.path.join(rir_path, "*/*/*.wav"))

    def additive_noise(self, noisecat, audio):

        clean_db = 10 * np.log10(np.mean(audio**2) + 1e-4)

        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(
            self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1])
        )

        noises = []

        for noise in noiselist:

            noiseaudio = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr = random.uniform(
                self.noisesnr[noisecat][0], self.noisesnr[noisecat][1]
            )
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2) + 1e-4)
            noises.append(
                np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio
            )

        return np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True) + audio

    def reverberate(self, audio):

        rir_file = random.choice(self.rir_files)

        rir, fs = soundfile.read(rir_file)
        rir = np.expand_dims(rir.astype(np.float), 0)
        rir = rir / np.sqrt(np.sum(rir**2))

        return signal.convolve(audio, rir, mode="full")[:, : self.max_audio]


class train_dataset_loader(Dataset):
    def __init__(
        self,
        train_list,
        augment,
        musan_path,
        rir_path,
        max_frames,
        size,
        val_list,
        train_path,
        val=False,
        **kwargs,
    ):

        self.augment_wav = AugmentWAV(
            musan_path=musan_path, rir_path=rir_path, max_frames=max_frames
        )

        self.train_list = val_list if val else train_list
        print(f"self.train_list: {self.train_list}")
        self.max_frames = max_frames
        self.musan_path = musan_path
        self.rir_path = rir_path
        self.augment = augment
        self.size = size

        # MHFA original
        # Read training files
        # with open(train_list) as dataset_file:
        #     lines = dataset_file.readlines()

        # Make a dictionary of ID names and ID indices
        # dictkeys = list(set([x.split()[0] for x in lines]))
        # dictkeys.sort()
        # dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
        #
        # # Parse the training list into file names and ID indices
        # self.data_list = []
        # self.data_label = []
        #
        # for lidx, line in enumerate(lines):
        #     data = line.strip().split()
        #
        #     speaker_label = dictkeys[data[0]]
        #     filename = os.path.join(train_path, data[1])
        #
        #     self.data_label.append(speaker_label)
        #     self.data_list.append(filename)

        # VK Manifests
        df = pd.read_csv(train_list)
        self.data_list = df["filepath"].tolist()
        self.data_label = df["label"].tolist()

    def __getitem__(self, indices):

        # feat = []
        # labels = []
        assert len(indices) == 1, f"Expected to get a single index."
        for index in indices:
            try:
                audio, mask = loadWAV(
                    self.data_list[index], self.max_frames, evalmode=False
                )
                audio = torch.torch.FloatTensor(audio).squeeze(0)
                mask = torch.torch.FloatTensor(mask).squeeze(0)
                label = torch.LongTensor([self.data_label[index]])
            except Exception as e:
                print(e)
                print(self.data_list[index])

            if len(audio.shape) == 3:
                print(self.data_list[index])
        return audio, mask, label

        #     if self.augment:
        #         augtype = random.randint(0, 5)
        #         if augtype == 0:
        #             audio = audio
        #         elif augtype == 1:
        #             audio = self.augment_wav.reverberate(audio)
        #         elif augtype == 2:
        #             audio = self.augment_wav.additive_noise("music", audio)
        #         elif augtype == 3:
        #             audio = self.augment_wav.additive_noise("speech", audio)
        #         elif augtype == 4:
        #             audio = self.augment_wav.additive_noise("noise", audio)
        #         elif augtype == 5:
        #             audio = self.augment_wav.additive_noise("speech", audio)
        #             audio = self.augment_wav.additive_noise("music", audio)
        #
        #     feat.append(audio)
        #
        # feat = np.concatenate(feat, axis=0)
        # print(f"feat: {feat}")
        # return torch.FloatTensor(feat), labels

    def __len__(self):
        return len(self.data_list) if self.size is None else self.size


class test_dataset_loader(Dataset):
    def __init__(self, test_list, eval_frames, num_eval, size, **kwargs):
        self.max_frames = eval_frames
        self.num_eval = num_eval
        self.test_list = test_list
        self.size = size

        df = pd.read_csv(test_list)
        self.audio_list = df["original_file"].tolist()
        self.audio2_list = df["compared_file"].tolist()
        self.data_label = df["labels"].tolist()

    def __getitem__(self, index):
        # print(self.test_list[index])
        audio, mask = loadWAV(
            self.audio_list[index],
            self.max_frames,
            evalmode=True,
            num_eval=self.num_eval,
        )

        audio2, mask2 = loadWAV(
            self.audio2_list[index],
            0,
            evalmode=True,
            num_eval=self.num_eval,
        )

        return (
            torch.FloatTensor(audio),
            torch.FloatTensor(audio2),
            self.data_label[index],
        )
        # return torch.FloatTensor(audio2), self.test_list[index]

    def __len__(self):
        return len(self.audio_list) if self.size is None else self.size


class train_dataset_sampler(torch.utils.data.Sampler):
    def __init__(
        self,
        data_source,
        nPerSpeaker,
        max_seg_per_spk,
        batch_size,
        distributed,
        seed,
        **kwargs,
    ):

        self.data_label = data_source.data_label
        self.data_list = data_source.data_list
        self.size = len(data_source)
        self.nPerSpeaker = nPerSpeaker
        self.max_seg_per_spk = max_seg_per_spk
        self.batch_size = batch_size
        self.epoch = 0
        self.seed = seed
        self.distributed = distributed

    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=g).tolist()

        data_dict = {}  # dict(zip(self.data_list, self.data_label))

        # Sort into dictionary of file indices for each ID
        for index in indices:
            speaker_label = self.data_label[index]
            if not (speaker_label in data_dict):
                data_dict[speaker_label] = []
            data_dict[speaker_label].append(index)

        ## Group file indices for each class
        dictkeys = list(data_dict.keys())
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i : i + sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        for findex, key in enumerate(dictkeys):
            data = data_dict[key]
            numSeg = round_down(min(len(data), self.max_seg_per_spk), self.nPerSpeaker)

            rp = lol(np.arange(numSeg), self.nPerSpeaker)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        ## Mix data in random order
        mixid = torch.randperm(len(flattened_label), generator=g).tolist()
        mixlabel = []
        mixmap = []

        ## Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = round_down(len(mixlabel), self.batch_size)
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        mixed_list = [flattened_list[i] for i in mixmap]

        ## Divide data to each GPU
        if self.distributed:
            total_size = round_down(
                len(mixed_list), self.batch_size * dist.get_world_size()
            )  # max(len(mixed_list),
            start_index = int((dist.get_rank()) / dist.get_world_size() * total_size)
            end_index = int((dist.get_rank() + 1) / dist.get_world_size() * total_size)
            self.num_samples = end_index - start_index
            return iter(mixed_list[start_index:end_index])
        else:
            total_size = round_down(
                len(mixed_list), self.batch_size
            )  # max(len(mixed_list),
            self.num_samples = total_size
            return iter(mixed_list[:total_size])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


if __name__ == "__main__":
    train_dataset = train_dataset_loader(
        train_list="/mnt/proj3/open-24-5/pengjy_new/WavLM_Adapter/CNCeleb_lst/CNCeleb_trainlist_200spk.txt",
        augment=False,
        musan_path="/mnt/proj3/open-24-5/pengjy_new/musan_split/",
        rir_path="/mnt/proj3/open-24-5/plchot/data_augment/16kHz/simulated_rirs/",
        max_frames=300,
        train_path="/mnt/proj3/open-24-5/pengjy_new/Data/CN-Celeb_flac/data",
    )

    train_sampler = train_dataset_sampler(
        train_dataset,
        nPerSpeaker=1,
        max_seg_per_spk=500,
        batch_size=100,
        distributed=False,
        seed=120,
    )
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=100,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
    )
    for data, data_label in train_loader:
        print(data.shape)
        data = data.transpose(1, 0)
        print(data.shape)
        quit()

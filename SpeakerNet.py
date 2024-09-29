#!/usr/bin/python
# -*- coding: utf-8 -*-

import importlib
import itertools
import math
import os
import pdb
import pickle
import random
import shutil
import sys
import time

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from DatasetLoader import test_dataset_loader
from metrics import get_eer, get_min_c
from tuneThreshold import tuneThresholdfromScore


def check_for_nans(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN detected in parameter: {name}")
        else:
            print(f"No NaNs in parameter: {name}")


class WrappedModel(nn.Module):

    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, data, mask, label=None, l2_reg_dict=None):
        return self.module(data, mask, label, l2_reg_dict)


class SpeakerNet(nn.Module):

    def __init__(self, model, optimizer, trainfunc, nPerSpeaker, **kwargs):
        super(SpeakerNet, self).__init__()

        SpeakerNetModel = importlib.import_module("models." + model).__getattribute__(
            "MainModel"
        )
        self.__S__ = SpeakerNetModel(**kwargs)

        LossFunction = importlib.import_module("loss." + trainfunc).__getattribute__(
            "LossFunction"
        )
        self.__L__ = LossFunction(**kwargs)

        self.nPerSpeaker = nPerSpeaker
        self.weight_finetuning_reg = kwargs["weight_finetuning_reg"]

    def forward(self, data, mask, label=None, l2_reg_dict=None):
        if label is None:
            # data_reshape = data[0].cuda()
            outp = self.__S__.forward(wav=data[0], padding_mask=mask)
            return outp
        else:
            # data_reshape = data[0].reshape(-1, data[0].size()[-1]).cuda()
            # print(f"data_reshape.shape: {data_reshape.shape}")
            # outp = self.__S__.forward([data_reshape, data[1]])
            outp = self.__S__.forward(wav=data[0], padding_mask=mask)
            nloss, prec1 = self.__L__.forward(outp, label)

            if l2_reg_dict is not None:
                Learned_dict = l2_reg_dict
                l2_reg = 0
                for name, param in self.__S__.model.named_parameters():
                    if name in Learned_dict:
                        l2_reg = l2_reg + torch.norm(
                            param - Learned_dict[name].cuda(), 2
                        )
                tloss = nloss / nloss.detach() + self.weight_finetuning_reg * l2_reg / (
                    l2_reg.detach() + 1e-5
                )
            else:
                tloss = nloss
                # print("Without L2 Reg")
            return tloss, prec1, nloss


class ModelTrainer(object):

    def __init__(self, speaker_model, optimizer, scheduler, gpu, mixedprec, **kwargs):

        self.__model__ = speaker_model

        WavLM_params = list(map(id, self.__model__.module.__S__.model.parameters()))
        Backend_params = filter(
            lambda p: id(p) not in WavLM_params, self.__model__.module.parameters()
        )
        self.path = kwargs["pretrained_model_path"]

        Optimizer = importlib.import_module("optimizer." + optimizer).__getattribute__(
            "Optimizer"
        )

        # Define the initial param groups
        param_groups = [{"params": Backend_params, "lr": kwargs["LR_MHFA"]}]

        # Extract the encoder layers
        encoder_layers = self.__model__.module.__S__.model.encoder.layers

        # Iterate over the encoder layers to create param groups
        for i in range(
            24
        ):  # Assuming 12 layers from 0 to 11 (for BASE model, when it comes to LARGE model, 12->24)
            lr = kwargs["LR_Transformer"] * (kwargs["LLRD_factor"] ** i)
            param_groups.append({"params": encoder_layers[i].parameters(), "lr": lr})

        # Initialize the optimizer with these param groups
        self.__optimizer__ = Optimizer(param_groups, **kwargs)

        # self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)

        Scheduler = importlib.import_module("scheduler." + scheduler).__getattribute__(
            "Scheduler"
        )
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

        # self.scaler = GradScaler()

        self.gpu = gpu

        self.mixedprec = mixedprec
        print("Mix prec: %s" % (self.mixedprec))

        assert self.lr_step in ["epoch", "iteration"]

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader, verbose):

        self.__model__.train()

        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss = 0
        top1 = 0  # EER or accuracy

        tstart = time.time()
        Learned_dict = {}
        checkpoint = torch.load(self.path)
        for name, param in checkpoint["model"].items():
            if "w2v_encoder.w2v_model." in name:
                newname = name.replace("w2v_encoder.w2v_model.", "")
            else:
                newname = name
            Learned_dict[newname] = param

        # for i, data in tqdm(
        #         enumerate(loader),
        #         total=len(loader),
        #         desc="Iterating through epoch...",
        #         leave=True,
        #         disable=not verbose
        # ):
        for i, data in enumerate(loader):
            # if i > 10:
            #     break
            # data = data.transpose(1, 0)
            self.__model__.zero_grad()
            # label = data_label[0].cuda() #[bs, 1]
            tloss, prec1, spkloss = self.__model__(
                data=[data[0].cuda(), "train"],
                mask=data[1].cuda(),
                label=data[2].cuda(),
                l2_reg_dict=Learned_dict,
            )  # data: [bs, n_samples]
            torch.nn.utils.clip_grad_norm_(self.__model__.parameters(), 1)
            tloss.backward()

            self.__optimizer__.step()

            loss += spkloss.detach().cpu()
            top1 += prec1.detach().cpu()

            counter += 1
            index += stepsize

            telapsed = time.time() - tstart
            tstart = time.time()

            if verbose:
                sys.stdout.write("\rProcessing (%d) " % (index))
                sys.stdout.write(
                    "Loss %f TEER/TAcc %2.3f%% - %.2f Hz "
                    % (loss / counter, top1 / counter, stepsize / telapsed)
                )
                sys.stdout.flush()

            if self.lr_step == "iteration":
                self.__scheduler__.step()

        if self.lr_step == "epoch":
            self.__scheduler__.step()

        sys.stdout.write("\n")

        return (loss / counter, top1 / counter)

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Evaluate network
    # ## ===== ===== ===== ===== ===== ===== ===== =====
    @torch.no_grad()
    def evaluate_network(self, loader, verbose):

        self.__model__.eval()
        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss = 0
        top1 = 0  # EER or accuracy

        tstart = time.time()
        Learned_dict = {}
        checkpoint = torch.load(self.path)
        for name, param in checkpoint["model"].items():
            if "w2v_encoder.w2v_model." in name:
                newname = name.replace("w2v_encoder.w2v_model.", "")
            else:
                newname = name
            Learned_dict[newname] = param

        # for i, data in tqdm(
        #         enumerate(loader),
        #         total=len(loader),
        #         desc="Iterating through validation epoch...",
        #         leave=True,
        #         disable=verbose
        # ):
        for i, data in enumerate(loader):
            nloss, prec1, spkloss = self.__model__(
                data=[data[0].cuda(), "train"],
                mask=data[1].cuda(),
                label=data[2].cuda(),
                l2_reg_dict=Learned_dict,
            )  # data: [bs, n_samples]
            loss += spkloss.detach().cpu()
            top1 += prec1.detach().cpu()

            counter += 1
            index += stepsize

            telapsed = time.time() - tstart
            tstart = time.time()

            if verbose:
                sys.stdout.write("\rProcessing validation (%d) " % (index))
                sys.stdout.write(
                    "Loss %f VEER/VAcc %2.3f%% - %.2f Hz "
                    % (loss / counter, top1 / counter, stepsize / telapsed)
                )
                sys.stdout.flush()

        sys.stdout.write("\n")

        return loss / counter, top1 / counter

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(
        self, test_list, val_list, mode, nDataLoaderThread, num_eval=5, **kwargs
    ):

        self.__model__.eval()

        test_dataset = test_dataset_loader(
            test_list if mode == "test" else val_list, num_eval=num_eval, **kwargs
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=nDataLoaderThread,
            drop_last=False,
        )
        print(f"Len of test loader: {len(test_loader)}")
        embeddings1 = []
        embeddings2 = []
        labels = []

        for idx, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            inp1 = data[0][0].cuda()
            inp2 = data[1][0].cuda()
            label = data[2]

            ref_feat = self.__model__(
                data=[inp1, "test"], mask=None, label=None, l2_reg_dict=None
            ).cuda()
            embeddings1.append(ref_feat.detach().cpu().numpy())
            ref_feat_2 = self.__model__(
                data=[inp2, "test"], mask=None, label=None, l2_reg_dict=None
            ).cuda()
            embeddings2.append(ref_feat_2.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())

        dcf = get_min_c(embeddings1, embeddings2, labels)
        eer = get_eer(embeddings1, embeddings2, labels)
        return eer, dcf

    def saveParameters(self, path):
        torch.save(self.__model__.module.state_dict(), path)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path, map_location="cuda:%d" % self.gpu)
        # loaded_state = torch.load(path, map_location="cpu");

        for name, param in loaded_state.items():
            origname = name

            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                print(
                    "Wrong parameter length: %s, model: %s, loaded: %s"
                    % (origname, self_state[name].size(), loaded_state[origname].size())
                )
                continue

            self_state[name].copy_(param)

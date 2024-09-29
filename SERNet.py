#!/usr/bin/python
# -*- coding: utf-8 -*-

import importlib
import random
import sys
import time

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score)

from DatasetLoader import test_dataset_loader


class WrappedModel(nn.Module):

    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None, l2_reg_dict=None, return_prediction=False):
        return self.module(
            data=x, l2_reg_dict=l2_reg_dict, return_prediction=return_prediction
        )


class SERNet(nn.Module):

    def __init__(self, model, optimizer, trainfunc, **kwargs):
        super(SERNet, self).__init__()

        SpeakerNetModel = importlib.import_module("models." + model).__getattribute__(
            "MainModel"
        )
        self.__S__ = SpeakerNetModel(**kwargs)

        LossFunction = importlib.import_module("loss." + trainfunc).__getattribute__(
            "LossFunction"
        )
        self.__L__ = LossFunction(**kwargs)

        self.weight_finetuning_reg = kwargs["weight_finetuning_reg"]

    def forward(self, data, l2_reg_dict, return_prediction):
        waveform = data["waveform"].to("cuda")
        padding_mask = data["padding_mask"].to("cuda")
        label = data["emotion"].to("cuda")
        outp = self.__S__.forward(waveform, padding_mask)
        loss = self.__L__.forward(outp, label)

        if l2_reg_dict is not None:
            Learned_dict = l2_reg_dict
            l2_reg = 0
            for name, param in self.__S__.model.named_parameters():
                if name in Learned_dict:
                    l2_reg = l2_reg + torch.norm(param - Learned_dict[name].cuda(), 2)
            tloss = loss / loss.detach() + self.weight_finetuning_reg * l2_reg / (
                l2_reg.detach() + 1e-5
            )
        else:
            tloss = loss
            # print("Without L2 Reg")

        if return_prediction:
            return tloss, loss, outp
        else:
            return tloss, loss


class ModelTrainer(object):

    def __init__(self, speaker_model, optimizer, scheduler, gpu, mixedprec, **kwargs):

        self.__model__ = speaker_model
        self.batch_size = kwargs.get("batch_size")
        WavLM_params = list(map(id, self.__model__.module.__S__.model.parameters()))
        Backend_params = filter(
            lambda p: id(p) not in WavLM_params, self.__model__.module.parameters()
        )
        backend_params_names = [
            n
            for n, p in self.__model__.module.named_parameters()
            if id(p) not in WavLM_params
        ]
        self.path = kwargs["pretrained_model_path"]
        self.accumulate_gradient_each_n_step = kwargs["accumulate_grad_each_n_step"]
        # print("Parameters in optim: \n",
        #     backend_params_names
        # )
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

        stepsize = self.batch_size
        counter = 0
        index = 0
        loss = 0

        tstart = time.time()
        Learned_dict = {}
        checkpoint = torch.load(self.path)
        for name, param in checkpoint["model"].items():
            if "w2v_encoder.w2v_model." in name:
                newname = name.replace("w2v_encoder.w2v_model.", "")
            else:
                newname = name
            Learned_dict[newname] = param
        torch.set_grad_enabled(True)
        self.__model__.zero_grad()
        for i, data in enumerate(loader):
            total_loss, ce_loss = self.__model__(
                data, Learned_dict, return_prediction=False
            )
            total_loss = total_loss / self.accumulate_gradient_each_n_step

            total_loss.backward()
            if i % self.accumulate_gradient_each_n_step == 0:
                # print(f"Did optimizer step!")
                self.__optimizer__.step()
                self.__model__.zero_grad()
                if self.lr_step == "iteration":
                    self.__scheduler__.step()

            # Ensure total_loss is a scalar
            if total_loss.dim() == 0:
                total_loss = total_loss.item()

            loss += total_loss
            counter += 1
            index += stepsize

            telapsed = time.time() - tstart
            tstart = time.time()

            if verbose:
                sys.stdout.write("\rProcessing (%d) " % i)
                sys.stdout.write(
                    "Loss %f - %.2f Hz " % (loss / counter, stepsize / telapsed)
                )
                sys.stdout.flush()

        if self.lr_step == "epoch":
            self.__scheduler__.step()

        sys.stdout.write("\n")

        return loss / counter

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    @torch.no_grad()
    def evaluate(self, loader, **kwargs):
        self.__model__.eval()

        stepsize = self.batch_size
        counter = 0
        index = 0
        loss = 0
        target_labels = []
        predicted_labels = []

        tstart = time.time()
        Learned_dict = {}
        checkpoint = torch.load(self.path)
        for name, param in checkpoint["model"].items():
            if "w2v_encoder.w2v_model." in name:
                newname = name.replace("w2v_encoder.w2v_model.", "")
            else:
                newname = name
            Learned_dict[newname] = param
        for i, data in enumerate(loader):
            total_loss, ce_loss, prediction_logits = self.__model__(
                data, Learned_dict, return_prediction=True
            )
            # Ensure total_loss is a scalar
            if total_loss.dim() == 0:
                total_loss = total_loss.item()
            predictions = (
                torch.argmax(F.softmax(prediction_logits, dim=1), dim=1)
                .detach()
                .cpu()
                .tolist()
            )
            targets = data["emotion"].tolist()
            predicted_labels.extend(predictions)
            target_labels.extend(targets)
            loss += total_loss
            counter += 1
            index += stepsize
            telapsed = time.time() - tstart

        assert len(target_labels) == len(predicted_labels), (
            f"Expected to gather equal amount of targets and " f"predictions."
        )
        # print(f"target_labels: {target_labels}")
        # print(f"predicted_labels: {predicted_labels}")
        balanced_accuracy = balanced_accuracy_score(target_labels, predicted_labels)
        accuracy = accuracy_score(target_labels, predicted_labels)
        f1_weighted = f1_score(
            target_labels, predicted_labels, average="weighted", zero_division=0
        )
        f1_macro = f1_score(
            target_labels, predicted_labels, average="macro", zero_division=0
        )
        precision_macro = precision_score(
            target_labels, predicted_labels, average="macro", zero_division=0
        )
        precision_weighted = precision_score(
            target_labels, predicted_labels, average="weighted", zero_division=0
        )
        return (
            balanced_accuracy,
            accuracy,
            f1_weighted,
            f1_macro,
            precision_macro,
            precision_weighted,
        )

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

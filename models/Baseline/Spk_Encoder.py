import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from .WavLM import *


class MHFA(nn.Module):
    def __init__(
        self,
        head_nb=8,
        inputs_dim=768,
        compression_dim=128,
        outputs_dim=256,
        n_layers=24,
    ):
        super(MHFA, self).__init__()

        # Define learnable weights for key and value computations across layers
        self.weights_k = nn.Parameter(data=torch.ones(n_layers + 1), requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(n_layers + 1), requires_grad=True)

        # Initialize given parameters
        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim

        # Define compression linear layers for keys and values
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)

        # Define linear layer to compute multi-head attention weights
        self.att_head = nn.Linear(self.cmp_dim, self.head_nb)

        # Define a fully connected layer for final output
        self.pooling_fc = nn.Linear(self.head_nb * self.cmp_dim, self.ous_dim)

    def forward(self, x, padding_mask=None):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]

        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(
            x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1
        ).transpose(1, 2)

        # Compute the value in a similar fashion
        v = torch.sum(
            x.mul(nn.functional.softmax(self.weights_v, dim=-1)), dim=-1
        ).transpose(1, 2)

        # Pass the keys and values through compression linear layers
        k = self.cmp_linear_k(k)
        v = self.cmp_linear_v(v)

        # Compute attention weights using compressed keys
        att_k = self.att_head(k)

        # Adjust dimensions for computing attention output
        v = v.unsqueeze(-2)
        padding_mask_expanded = padding_mask.unsqueeze(-1).expand_as(att_k)
        att_k_masked = att_k.masked_fill(padding_mask_expanded, -np.inf)

        # Compute attention output by taking weighted sum of values using softmaxed attention weights
        pooling_outs = torch.sum(
            v.mul(nn.functional.softmax(att_k_masked, dim=1).unsqueeze(-1)), dim=1
        )

        # Reshape the tensor before passing through the fully connected layer
        b, h, f = pooling_outs.shape
        pooling_outs = pooling_outs.reshape(b, -1)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(pooling_outs)

        return outs


class spk_extractor(nn.Module):
    def __init__(self, **kwargs):
        super(spk_extractor, self).__init__()
        print("Pre-trained Model: {}".format(kwargs["pretrained_model_path"]))
        checkpoint = torch.load(kwargs["pretrained_model_path"])
        cfg = WavLMConfig(checkpoint["cfg"])
        self.cfg = checkpoint["cfg"]
        self.model = WavLM(cfg)
        self.loadParameters(checkpoint["model"])
        self.backend = MHFA(
            head_nb=kwargs["head_nb"],
            inputs_dim=self.cfg["encoder_embed_dim"],
            n_layers=self.cfg["encoder_layers"],
            outputs_dim=kwargs["nClasses"],
        )

    def forward(self, wav, padding_mask=None):

        x = wav
        cnn_outs, layer_results = self.model.extract_features(
            x, output_layer=self.cfg["encoder_layers"], padding_mask=padding_mask
        )
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
        x = torch.stack(layer_reps).transpose(0, -1).transpose(0, 1)
        if padding_mask is not None:
            padding_mask_wav_lm = self.model.padding_mask
        else:
            padding_mask_wav_lm = None

        out = self.backend(x, padding_mask_wav_lm)
        return out

    def loadParameters(self, param):

        self_state = self.model.state_dict()
        loaded_state = param

        for name, param in loaded_state.items():
            origname = name

            if name not in self_state:
                # print("%s is not in the model."%origname);
                continue

            if self_state[name].size() != loaded_state[origname].size():
                print(
                    "Wrong parameter length: %s, model: %s, loaded: %s"
                    % (origname, self_state[name].size(), loaded_state[origname].size())
                )
                continue

            self_state[name].copy_(param)


def MainModel(**kwargs):
    model = spk_extractor(**kwargs)
    return model

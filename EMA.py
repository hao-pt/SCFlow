# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

"""
Codes adapted from https://github.com/NVlabs/LSGM/blob/main/util/ema.py
"""

import copy
import warnings

import torch
from torch.optim import Optimizer


class EMA(Optimizer):
    def __init__(self, opt, ema_decay):
        self.ema_decay = ema_decay
        self.apply_ema = self.ema_decay > 0.0
        self.optimizer = opt
        self.state = opt.state
        self.param_groups = opt.param_groups

    def step(self, *args, **kwargs):
        retval = self.optimizer.step(*args, **kwargs)

        # stop here if we are not applying EMA
        if not self.apply_ema:
            return retval

        ema, params = {}, {}
        for group in self.optimizer.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                state = self.optimizer.state[p]

                # State initialization
                if "ema" not in state:
                    state["ema"] = p.data.clone()

                if p.shape not in params:
                    params[p.shape] = {"idx": 0, "data": []}
                    ema[p.shape] = []

                params[p.shape]["data"].append(p.data)
                ema[p.shape].append(state["ema"])

            for i in params:
                params[i]["data"] = torch.stack(params[i]["data"], dim=0)
                ema[i] = torch.stack(ema[i], dim=0)
                ema[i].mul_(self.ema_decay).add_(
                    params[i]["data"], alpha=1.0 - self.ema_decay,
                )

            for p in group["params"]:
                if p.grad is None:
                    continue
                idx = params[p.shape]["idx"]
                self.optimizer.state[p]["ema"] = ema[p.shape][idx, :]
                params[p.shape]["idx"] += 1

        return retval

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # load_state_dict loads the data to self.state and self.param_groups. We need to pass this data to
        # the underlying optimizer too.
        self.optimizer.state = self.state
        self.optimizer.param_groups = self.param_groups

    def swap_parameters_with_ema(self, store_params_in_ema):
        """This function swaps parameters with their ema values. It records original parameters in the ema
        parameters, if store_params_in_ema is true."""

        # stop here if we are not applying EMA
        if not self.apply_ema:
            warnings.warn(
                "swap_parameters_with_ema was called when there is no EMA weights.",
            )
            return

        for group in self.optimizer.param_groups:
            for i, p in enumerate(group["params"]):
                if not p.requires_grad:
                    continue
                ema = self.optimizer.state[p]["ema"]
                if store_params_in_ema:
                    tmp = p.data.detach()
                    p.data = ema.detach()
                    self.optimizer.state[p]["ema"] = tmp
                else:
                    p.data = ema.detach()


class EMAMODEL:
    def __init__(self, model):
        self.ema_model = copy.deepcopy(model)
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)
        self.ema_model.eval()

    @torch.no_grad()
    def ema_step(self, decay_rate=0.9999, model=None):
        for param, ema_param in zip(model.parameters(), self.ema_model.parameters()):
            ema_param.data.mul_(decay_rate).add_(param.data, alpha=1.0 - decay_rate)

    @torch.no_grad()
    def ema_swap(self, model=None):
        for param, ema_param in zip(self.ema_model.parameters(), model.parameters()):
            tmp = param.data.detach()
            param.data = ema_param.detach()
            ema_param.data = tmp

    @torch.no_grad()
    def __call__(self, t, z, **kwargs):
        return self.ema_model(t, z, **kwargs)

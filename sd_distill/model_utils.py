import copy

import numpy as np
import torch


def alpha(t):
    # DDPM defines x_t(x, z) = alpha(t)x + sqrt(1 - alpha(t)^2)z
    a = 19.9
    b = 0.1
    exp = torch.exp if isinstance(t, torch.Tensor) else np.exp
    return exp(-0.25 * a * t**2 - 0.5 * b * t)


def dalpha_dt(t):
    a = 19.9
    b = 0.1
    alpha_t = alpha(t)
    return (-0.5 * a * t - 0.5 * b) * alpha_t


def d_1_minus_alpha_sq_dt(t):
    alpha_t = alpha(t)
    return 0.5 * (1 - alpha_t**2) ** (-0.5) * (-2 * alpha_t) * dalpha_dt(t)


def create_ema_and_scales_fn(
    target_ema_mode,
    start_ema,
    scale_mode,
    start_scales,
    end_scales,
    total_steps,
    distill_steps_per_iter=5000,
):
    def ema_and_scales_fn(step):
        if target_ema_mode == "fixed" and scale_mode == "fixed":
            target_ema = start_ema
            scales = start_scales
        elif target_ema_mode == "fixed" and scale_mode == "progressive":
            target_ema = start_ema
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2,
                )
                - 1,
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            scales = scales + 1

        elif target_ema_mode == "adaptive" and scale_mode == "progressive":
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2,
                )
                - 1,
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            c = -np.log(start_ema) * start_scales
            target_ema = np.exp(-c / scales)
            scales = scales + 1
        elif target_ema_mode == "fixed" and scale_mode == "progdist":
            distill_stage = step // distill_steps_per_iter
            scales = start_scales // (2**distill_stage)
            scales = np.maximum(scales, 2)

            sub_stage = np.maximum(
                step - distill_steps_per_iter * (np.log2(start_scales) - 1),
                0,
            )
            sub_stage = sub_stage // (distill_steps_per_iter * 2)
            sub_scales = 2 // (2**sub_stage)
            sub_scales = np.maximum(sub_scales, 1)

            scales = np.where(scales == 2, sub_scales, scales)

            target_ema = 1.0
        else:
            raise NotImplementedError

        return float(target_ema), int(scales)

    return ema_and_scales_fn


class LossRecord:
    def __init__(self, register_key=["fm_losses", "con_losses", "rf_losses"]):
        self.times = np.array([])
        self.register_key = register_key
        self.losses = {}
        for key in register_key:
            self.losses[key] = np.array([])

    def reset(self):
        self.times = np.array([])
        for key in self.register_key:
            self.losses[key] = np.array([])

    def plot(self, save_path):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(20, 5))
        order = np.argsort(self.times)
        for key in self.register_key:
            plt.plot(self.times[order], self.losses[key][order], label=key)
        plt.xlabel("times")
        plt.ylabel("losses")
        plt.yscale("log")
        plt.legend(loc="lower left")
        plt.savefig(save_path)

    def add(self, t, track_loss):
        # t = deepcopy(t)
        # track_loss = deepcopy(track_loss)
        t = t.detach().float().cpu().numpy()
        t = t.squeeze()
        for key in self.register_key:
            self.losses[key] = np.concatenate(
                (
                    self.losses[key],
                    track_loss[key].detach().squeeze().float().cpu().numpy(),
                ),
            )
        self.times = np.concatenate((self.times, t.reshape(-1)))


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

import numpy as np
import torch as th

def expand_t(t, x_shape):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x_shape: [batch_dim,...], data shape
    """
    dims = [1] * (len(x_shape) - 1)
    t = t.view(t.size(0), *dims)
    return t

def compute_gamma(t, x_shape, form="none"):
    """Compute the diffusion term of the SDE
    Args:
        t: [batch_dim,], time vector
        x_shape: [batch_dim,], data shape
        form: str, form of the gamma term
    """
    t = expand_t(t, x_shape)
    choices = {
        "none": th.zeros((1,), device=t.device),
        "constant": th.full((1,), 1.0, device=t.device),
        "log": th.log(t - t**2 + 1),
        "sin": th.sin(np.pi * t) ** 2,
        # "SBDM": 2. * self.compute_drift(x, t)[1],
        # "sigma": self.compute_sigma_t(t)[0],
        # "linear": (1 - t),
        # "decreasing": 0.25 * (th.cos(np.pi * t) + 1) ** 2,
    }

    try:
        diffusion = choices[form]
    except KeyError:
        raise NotImplementedError(f"Gamma {form} not implemented")

    return diffusion


def compute_d_gamma(t, x_shape, form="constant"):
    """Compute the derivative of gamma term
    Args:
        t: [batch_dim,], time vector
        x_shape: [batch_dim,], data shape
        form: str, form of the diffusion term
        norm: float, norm of the diffusion term
    """
    t = expand_t(t, x_shape)
    choices = {
        "none": th.zeros((1,), device=t.device),
        "constant": th.zeros((1,), device=t.device),
        "log": (1 - 2*t)/(t - t**2 + 1),
        "sin": 2 * np.pi * th.sin(np.pi * t) * th.cos(np.pi * t), # norm * th.sin(np.pi * t) ** 2,
        # "SBDM": norm * 2. * self.compute_drift(x, t)[1],
        # "sigma": norm * self.compute_sigma_t(t)[0],
        # "linear": th.full((1,), -norm),
        # "decreasing": -0.5 * np.pi * norm * th.sin(np.pi * t) * (norm * th.cos(np.pi * t) + 1), # 0.25 * (norm * th.cos(np.pi * t) + 1) ** 2,
    }

    try:
        diffusion = choices[form]
    except KeyError:
        raise NotImplementedError(f"Gamma {form} not implemented")

    return diffusion


def stochastic_forward(xt, u, t, form="none"):
    perturbation = th.randn_like(xt)
    xt = xt + compute_gamma(t, xt.shape, form) * perturbation
    ut = u # + compute_d_gamma(t, xt.shape, form) * perturbation
    return xt, ut, t


def sigmoid_schedule(t, start=-3, end=3, tau=1.0, clip_min=1e-9):
    def sigmoid(x):
        return 1./(1 + np.exp(-x))
    # A gamma function based on sigmoid function.
    v_start = sigmoid(start / tau)
    v_end = sigmoid(end / tau)
    output = sigmoid((t * (end - start) + start) / tau)
    output = (v_end - output) / (v_end - v_start)
    return th.clamp(output, clip_min, 1.)

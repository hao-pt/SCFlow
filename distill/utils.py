import torch

def alpha(t):
    # DDPM defines x_t(x, z) = alpha(t)x + sqrt(1 - alpha(t)^2)z
    a = 19.9
    b = 0.1
    exp = torch.exp if isinstance(t, torch.Tensor) else np.exp
    return exp(-0.25 * a * t ** 2 - 0.5 * b * t)

def dalpha_dt(t):
    a = 19.9
    b = 0.1
    alpha_t = alpha(t)
    return (-0.5 * a * t - 0.5 * b) * alpha_t

def d_1_minus_alpha_sq_dt(t):
    a = 19.9
    b = 0.1
    alpha_t = alpha(t)
    return 0.5 * (1 - alpha_t ** 2) ** (-0.5) * (-2 * alpha_t) * dalpha_dt(t)
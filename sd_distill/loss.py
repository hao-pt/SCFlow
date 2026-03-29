import torch


def grad_penalty_call(args, D_real, x_t):
    grad_real = torch.autograd.grad(
        outputs=D_real.sum(),
        inputs=x_t,
        create_graph=True,
        only_inputs=True,
    )[0]
    grad_penalty = grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2

    grad_penalty = args.r1_gamma / 2 * grad_penalty
    return grad_penalty


def huber_loss(x, y, c=0.01):
    c = torch.tensor(c, device=x.device)
    # return (torch.sqrt(torch.mean((x - y)**2., dim=(1,2,3)) + c**2) - c)
    # return (torch.sqrt(torch.sqrt((x - y)**2. + 1e-5) + c**2.) - c)
    return torch.sqrt((x - y) ** 2.0 + c**2) - c  # .mean(dim=(1,2,3)

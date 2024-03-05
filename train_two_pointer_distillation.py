# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import shutil
import argparse
from functools import partial
from omegaconf import OmegaConf
from time import time
import logging
import copy
from tqdm import tqdm
import numpy as np
import torch
from torchdiffeq import odeint_adjoint as odeint
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.distributed as dist
from torch.multiprocessing import Process
from datasets_prep import get_dataset
from models import create_network, create_discriminator
from EMA import EMA, EMAMODEL
from sampler.karras_sample import karras_sample
from distill.flows import ConsistencyFlow
import torch.fft as fft


class LossRecord:
    def __init__(self, register_key = ["fm_losses", "con_losses", "rf_losses"]):
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
        from copy import deepcopy
        # t = deepcopy(t)
        # track_loss = deepcopy(track_loss)
        t = t.detach().cpu().numpy()
        t = t.squeeze()
        for key in self.register_key:
            self.losses[key] = np.concatenate((self.losses[key], track_loss[key].detach().squeeze().cpu().numpy())) 
        self.times = np.concatenate((self.times, t))
        
def batch_mse(input, target):
    return torch.mean((input - target)**2, dim=(1,2,3))


def frequency_weighted_loss(predicted, target, high_weight=1.0, low_weight=0.0, cutoff = 10, scale_high = 0.5):
    # Compute Fourier Transforms
    predicted_fft = fft.fftn(predicted, dim=(-2, -1))
    target_fft = fft.fftn(target, dim=(-2, -1))

    # Compute magnitude spectra
    predicted_mag = torch.abs(predicted_fft)
    target_mag = torch.abs(target_fft)

    # Create frequency masks for high and low frequencies
    high_mask = torch.ones_like(predicted_mag)
    low_mask = torch.ones_like(predicted_mag)

    # Modify the masks to give higher weight to high frequencies
    high_mask[..., cutoff:, :] = high_weight
    high_mask[..., :, cutoff:] = high_weight

    # Modify the masks to give higher weight to low frequencies
    low_mask[..., :cutoff, :] = low_weight
    low_mask[..., :, :cutoff] = low_weight

    # Apply the masks to the magnitude spectra
    weighted_predicted_mag = predicted_mag * high_mask
    weighted_target_mag = target_mag * high_mask

    # Calculate the mean squared error loss
    loss_high = F.mse_loss(weighted_predicted_mag, weighted_target_mag)

    # Apply the masks to the magnitude spectra for low frequencies
    weighted_predicted_mag_low = predicted_mag * low_mask
    weighted_target_mag_low = target_mag * low_mask

    # Calculate the mean squared error loss for low frequencies
    loss_low = F.mse_loss(weighted_predicted_mag_low, weighted_target_mag_low)

    # Combine the losses with a custom weighting (you may adjust this based on your needs)
    total_loss = scale_high * loss_high + (1-scale_high) * loss_low

    return total_loss


class FocalFrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight

def huber_loss(x, y, c = 0.01):
    c = torch.tensor(c, device=x.device)
    return torch.sqrt(batch_mse(x, y) + c**2) - c


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def get_weight(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def sample_ode_generative(model, z1, N, use_tqdm=True, solver = 'euler', model_kwargs={}):
    assert solver in ['euler', 'heun']
    tq = tqdm if use_tqdm else lambda x: x
    if solver == 'heun' and N % 2 == 0:
        raise ValueError("N must be odd when using Heun's method.")
    if solver == 'heun':
        N = (N + 1) // 2
    dt = -1./N

    traj = [] # to store the trajectory
    x0hat_list = []
    z = z1.detach().clone()
    batchsize = z.shape[0]
    traj.append(z.detach().clone())

    for i in tq(reversed(range(1,N+1))):
        t = torch.ones((batchsize,1), device=z1.device) * i / N
        t_next = torch.ones((batchsize,1), device=z1.device) * (i-1) / N
        vt = model(t.squeeze(), z, **model_kwargs)
        if solver == 'heun' and i > 1:
            z_next = z.detach().clone() + vt * dt
            vt_next = model(t_next.squeeze(), z_next, **model_kwargs)
            vt = (vt + vt_next) / 2
        x0hat = z - vt * t.view(-1,1,1,1)
        x0hat_list.append(x0hat)
        z = z.detach().clone() + vt * dt
        traj.append(z.detach().clone())

    return traj, x0hat_list
#%%
def train(rank, gpu, args):
    from diffusers.models import AutoencoderKL
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    device = torch.device('cuda:{}'.format(gpu))
    dtype = torch.float16 if args.use_fp16 else torch.float32

    exp = args.exp
    parent_dir = "./saved_info/2_pointers_consistency/{}".format(args.dataset)
    
    
    exp_path = os.path.join(parent_dir, exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            config_dict = vars(args)
            OmegaConf.save(config_dict, os.path.join(exp_path, "config.yaml"))

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(), 
                logging.FileHandler(f"{exp_path}/log.txt")]
        )
        # Creating an object
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info(f"Exp path: {exp_path}")

    batch_size = args.batch_size
    
    model = create_network(args).to(device, dtype=dtype)
    if args.use_grad_checkpointing and "DiT" in args.model_type:
        model.set_gradient_checkpointing()
    
    first_stage_model = AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt).to(device, dtype=dtype)
    first_stage_model = first_stage_model.eval()
    first_stage_model.train = False
    for param in first_stage_model.parameters():
        param.requires_grad = False
    
    if rank == 0:
        logger.info('AutoKL size: {:.3f}MB'.format(get_weight(first_stage_model)))
        logger.info('FM size: {:.3f}MB'.format(get_weight(model)))
    
    broadcast_params(model.parameters())
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=False)
    model.train() 
    

    if args.use_ema:
        optimizer = EMA(optimizer, ema_decay=args.ema_decay)

    
    ckpt = torch.load(args.model_ckpt, map_location=device)
    model.load_state_dict(ckpt)
    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        model.load_state_dict(checkpoint['model_dict'])
        # load model
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        global_step = checkpoint["global_step"]
        del checkpoint
    else:
        global_step, init_iter = 0, 0


    teacher = copy.deepcopy(model)
    teacher.eval()
    assistant = copy.deepcopy(model)
    use_label = True if "imagenet" in args.dataset else False
    start_time = time()
    latent_res = args.image_size // args.f
    focal_loss = FocalFrequencyLoss()
    delta_t = torch.tensor(0., device=device).float()
    epsilon = torch.tensor(args.epsilon, device=device)
    logger.info("epsilon t is {}".format(delta_t))
    for iter in tqdm(range(init_iter+1, args.n_iter+1)):
        # reset record
        # loss_record.reset()
        z1 = torch.randn((batch_size, args.num_in_channels, latent_res, latent_res), device=device)
        optimizer.zero_grad()
        model_kwargs = {}
        with torch.no_grad():
            v_assistant = assistant(torch.ones((batch_size,)), z1, **model_kwargs).detach()
            zt_assistant = z1 - v_assistant * delta_t.view(-1, 1, 1, 1)
            v_teacher = teacher(1 - delta_t, zt_assistant, **model_kwargs).detach()
            zt_teacher = zt_assistant - v_teacher * (epsilon/2).view(-1, 1, 1, 1)
            v_teacher_next = teacher(1 - delta_t - epsilon/2, zt_teacher, **model_kwargs).detach()
            v_teacher = (v_teacher + v_teacher_next)/2
            z0_teacher = zt_assistant - v_teacher * (1 - delta_t).view(-1, 1, 1, 1)
            
            # x0_teacher = first_stage_model.decode(z0_teacher / args.scale_factor).sample
            # torchvision.utils.save_image(x0_teacher, "x0_teacher.jpg", normalize=True)
        
        # t = 1 - torch.rand((z1.shape[0],), device=device).float()*(delta_t + epsilon)
        # zt = z0_teacher * (1 - t).view(-1, 1, 1, 1) + torch.randn_like(z1) * t.view(-1, 1, 1, 1)
        # v_student = model(t, zt, **model_kwargs)
        # z0_student = zt - v_student * t.view(-1, 1, 1, 1)
        # v_student_ = model(torch.ones((batch_size,)), z1, **model_kwargs)
        # z0_student_ = z1 - v_student_
        # zt_ = z0_student_.detach() * (1 - t).view(-1, 1, 1, 1) + z1 * t.view(-1, 1, 1, 1)
        # vt_student_ = model(t, zt_, **model_kwargs)
        # z0t_student_ = zt_ - vt_student_ * t.view(-1, 1, 1, 1)
        # loss = huber_loss(z0_student, z0_teacher).mean() + huber_loss(z0_student_, z0_teacher).mean() # + huber_loss(z0_student_, z0_teacher).mean() + huber_loss(z0t_student_, z0_teacher).mean()
        # exit(0)
        # optimize
        t = 1 - torch.rand((z1.shape[0],), device=device).float()*(delta_t + epsilon)
        t = torch.clamp(t, 0, 1)
        v_student = model(torch.ones((batch_size,)), z1, **model_kwargs)
        z0_student = z1 - v_student
        zt =  z0_student.detach() * (1 - t).view(-1, 1, 1, 1) + z1 * t.view(-1, 1, 1, 1)
        vt_student = model(t, zt, **model_kwargs)
        z0t_student = zt - vt_student * t.view(-1, 1, 1, 1)
        # weight = lambda t: torch.exp(t**3)
        loss = torch.mean(huber_loss(z0t_student, z0_teacher)) + torch.mean(huber_loss(z0_student, z0_teacher))
        
        loss.backward()
        optimizer.step()
              
        if iter % 100 == 0:
            if rank == 0:
                # Measure training speed:
                end_time = time()
                steps_per_sec = 100 / (end_time - start_time)
                logger.info('Iteration{}, Loss: {}, Train Steps/Sec: {:.2f}'.format(
                    iter, 
                    loss.item(),
                    steps_per_sec))
                start_time = time()

        if rank == 0:
            if iter % args.plot_every == 0:
                with torch.no_grad():
                    rand = torch.randn_like(z1)[:4]
                    sample_model = partial(model, y=None)
                    traj, x0_list = sample_ode_generative(model, rand, args.num_sample_timesteps)
                    fake_image = traj[-1]
                    fake_image = first_stage_model.decode(fake_image / args.scale_factor).sample
                # x0_list = x0_list[::2]
                x0_list = torch.cat(x0_list, dim=0)
                x0_list = first_stage_model.decode(x0_list / args.scale_factor).sample
                torchvision.utils.save_image(x0_list, os.path.join(exp_path, 'x0_iter_{}.png'.format(iter)), normalize=True, nrow=4)
                torchvision.utils.save_image(fake_image, os.path.join(exp_path, 'image_iter_{}.png'.format(iter)), normalize=True)
                logger.info("Finish sampling")

            if iter % args.save_content_every == 0:
                logger.info('Saving content.')
                content = {'iter': iter + 1, 'args': args,
                        'model_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(content, os.path.join(exp_path, 'content.pth'))

            if iter % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizer.swap_parameters_with_ema(store_params_in_ema=True)
                torch.save(model.state_dict(), os.path.join(exp_path, 'model_{}.pth'.format(iter)))
                if args.use_ema:
                    optimizer.swap_parameters_with_ema(store_params_in_ema=True)
                    
        if rank == 0:
            if iter % args.n_iter_inner == 0:
                logger.info("Add epsilon to delta t")
                delta_t += epsilon
                delta_t = torch.clamp(delta_t, 0, 1)
                logger.info("Epsilon t is {}".format(delta_t))
                if delta_t >= 1:
                    logger.info("Delta t is so big. Break")
                    exit(0)
                assistant = copy.deepcopy(model)


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()


def cleanup():
    dist.destroy_process_group()


#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')

    parser.add_argument('--resume', action='store_true',default=False)
    parser.add_argument('--model_ckpt', type=str, default=None,
                            help="Model ckpt to init from")
    parser.add_argument('--model_type', type=str, default="adm",
                            help='model_type', choices=['adm', 'ncsn++', 'ddpm++', 'DiT-B/2', 'DiT-L/2', 'DiT-L/4', 'DiT-XL/2'])
    parser.add_argument('--image_size', type=int, default=256,
                            help='size of image')
    parser.add_argument('--f', type=int, default=8,
                            help='downsample rate of input image by the autoencoder')
    parser.add_argument('--scale_factor', type=float, default=0.18215,
                            help='size of image')
    parser.add_argument('--num_in_channels', type=int, default=4,
                            help='in channel image')
    parser.add_argument('--num_out_channels', type=int, default=4,
                            help='in channel image')
    parser.add_argument('--nf', type=int, default=256,
                            help='channel of model')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', nargs='+', type=int, default=(16,8),
                            help='resolution of applying attention')
    parser.add_argument('--ch_mult', nargs='+', type=int, default=(1,2,2,2),
                            help='channel mult')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--label_dim', type=int, default=0,
                            help='label dimension, 0 if unconditional')
    parser.add_argument('--augment_dim', type=int, default=0,
                            help='dimension of augmented label, 0 if not used')
    parser.add_argument('--num_classes', type=int, default=None,
                            help='num classes')
    parser.add_argument('--label_dropout', type=float, default=0.,
                            help='Dropout probability of class labels for classifier-free guidance')

    # Original ADM
    parser.add_argument('--layout', action='store_true')
    parser.add_argument('--use_origin_adm', action='store_true', default=True)
    parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
    parser.add_argument("--resblock_updown", type=bool, default=False)
    parser.add_argument("--use_new_attention_order", type=bool, default=False)
    parser.add_argument('--centered', action='store_false', default=True, help='-1,1 scale')
    parser.add_argument("--resamp_with_conv", type=bool, default=True)
    parser.add_argument('--num_heads', type=int, default=4, help='number of head')
    parser.add_argument('--num_head_upsample', type=int, default=-1, help='number of head upsample')
    parser.add_argument('--num_head_channels', type=int, default=-1, help='number of head channels')
    parser.add_argument('--pretrained_autoencoder_ckpt', type=str, default="stabilityai/sd-vae-ft-mse")

    # distill
    parser.add_argument('--num_sample_timesteps', type=int, default=10)

    # training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--datadir', default='./data')
    parser.add_argument('--use_fp16', action='store_true', default=False)
    parser.add_argument('--use_grad_checkpointing', action='store_true', default=False, help="Enable gradient checkpointing for mem saving")
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate g')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam')
    parser.add_argument('--lr_decay',action='store_true', default=False)
    parser.add_argument('--use_ema', action='store_true', default=False, help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
    parser.add_argument('--target_ema_decay', type=float, default=0.95, help='decay rate for target EMA model')

    # 2 students setting
    parser.add_argument('--epsilon', type=float, default=0.01, help='jump epsilon for distillation')
    parser.add_argument('--n_iter_inner', type=int, default=10000, help='jump epsilon for distillation')
    parser.add_argument('--n_iter', type=int, default=500000, help='jump epsilon for distillation')
    

    # saving
    parser.add_argument('--save_content', action='store_true', default=False)
    parser.add_argument('--save_content_every', type=int, default=10000, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=1000, help='save ckpt every x epochs')
    parser.add_argument('--plot_every', type=int, default=100, help='plot every x epochs')

    ###ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--master_port', type=str, default='6000',
                        help='port for master')

    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        try:
            torch.multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass
        
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print('starting in debug mode')

        init_processes(0, size, train, args)

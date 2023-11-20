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


def huber_loss(x, y, c = 0.01):
    return torch.sqrt(F.mse_loss(x, y) + c**2) - c

def grad_penalty_call(args, D_real, x_t):
    grad_real = torch.autograd.grad(
        outputs=D_real.sum(), inputs=x_t, create_graph=True
    )[0]
    grad_penalty = (
        grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
    ).mean()

    grad_penalty = args.r1_gamma / 2 * grad_penalty
    return grad_penalty


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


#%%
def train(rank, gpu, args):
    from diffusers.models import AutoencoderKL
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    device = torch.device('cuda:{}'.format(gpu))
    dtype = torch.float16 if args.use_fp16 else torch.float32

    exp = args.exp
    if args.model_ckpt:
        parent_dir = "./saved_info/reflow_consistency/{}".format(args.dataset)
    else:
        parent_dir = "./saved_info/flow_consistency/{}".format(args.dataset)
    
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
    dataset = get_dataset(args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=True)

    model = create_network(args).to(device, dtype=dtype)
    if args.use_grad_checkpointing and "DiT" in args.model_type:
        model.set_gradient_checkpointing()
    
    first_stage_model = AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt).to(device, dtype=dtype)
    first_stage_model = torch.compile(first_stage_model, mode="max-autotune")
    first_stage_model = first_stage_model.eval()
    first_stage_model.train = False
    for param in first_stage_model.parameters():
        param.requires_grad = False
    if args.use_gan:
        modelD = create_discriminator(args).to(device, dtype=dtype)
    
    if rank == 0:
        logger.info('AutoKL size: {:.3f}MB'.format(get_weight(first_stage_model)))
        logger.info('FM size: {:.3f}MB'.format(get_weight(model)))
        if args.use_gan:
            logger.info('Discriminator size: {:.3f}MB'.format(get_weight(modelD)))

    broadcast_params(model.parameters())
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, eta_min=1e-5)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=False)
    model.train() 
    if args.use_gan:
        broadcast_params(modelD.parameters())
        optimizerD = optim.AdamW(modelD.parameters(), lr=args.lrD, weight_decay=0.0)
        schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.num_epoch, eta_min=1e-5)
        modelD = nn.parallel.DistributedDataParallel(modelD, device_ids=[gpu], find_unused_parameters=True)
        modelD.train()
        modelD_ = modelD
        # modelD_ = torch.compile(modelD)

    if args.use_ema:
        optimizer = EMA(optimizer, ema_decay=args.ema_decay)

    if args.model_ckpt:
        ckpt = torch.load(args.model_ckpt, map_location=device)
        model.load_state_dict(ckpt)
    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        model.load_state_dict(checkpoint['model_dict'])
        # load G
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        global_step = checkpoint["global_step"]
        # load D
        if args.use_gan:
            modelD.load_state_dict(checkpoint['modelD_dict'])
            optimizerD.load_state_dict(checkpoint['optimizerD'])
            schedulerD.load_state_dict(checkpoint['schedulerD'])
        if rank == 0:
            logger.info("=> resume checkpoint (epoch {})"
                    .format(checkpoint['epoch']))
        del checkpoint
    else:
        global_step, epoch, init_epoch = 0, 0, 0


    teacher = copy.deepcopy(model)
    teacher.eval()
    target_ema = EMAMODEL(model)
    model_ = model
    model_ = torch.compile(model)
    
    flow = ConsistencyFlow(device, 
                            model=model_, 
                            ema_model=target_ema, 
                            threshold=args.init_threshold, 
                            pretrained_model=teacher)
    use_label = True if "imagenet" in args.dataset else False
    start_time = time()
    for epoch in range(init_epoch, args.num_epoch+1):
        train_sampler.set_epoch(epoch)

        for iteration, (x, y) in enumerate(tqdm(data_loader)):
            x0 = x.to(device, dtype=dtype, non_blocking=True)
            y = None if not use_label else y.to(device, non_blocking=True)
            z0 = first_stage_model.encode(x0).latent_dist.sample().mul_(args.scale_factor)
            # set grad for gan
            z0.requires_grad = True
            x0.requires_grad = True
            z1 = torch.randn_like(z0)
            t = torch.rand((z1.shape[0], 1), device=device).float()
            model_kwargs = {}
            _, _, _, curr_z0, post_z0, reflow_z0_rescon, reflow_z0 = flow.get_train_tuple(z0, 
                                                                                        z1, 
                                                                                        t=t, 
                                                                                        model_kwargs=model_kwargs)
            # gan part
            if args.use_gan:
                for p in modelD_.parameters():
                    p.requires_grad = True
                optimizerD.zero_grad()
                # calc D real
                Dreal = modelD_(z0, c=y)
                Dreal_loss = F.softplus(-Dreal).mean()
                # penalty discriminator
                grad_real = 0.
                if args.lazy_reg is None:
                    grad_real = grad_penalty_call(args, Dreal, z0)
                else:
                    if global_step % args.lazy_reg == 0:
                        grad_real = grad_penalty_call(args, Dreal, z0)
                Dfake = modelD_(curr_z0.detach().requires_grad_(), c=y) + modelD_(reflow_z0_rescon.detach().requires_grad_(), c=y)
                Dfake_loss = F.softplus(Dfake).mean()
                Dloss = Dreal_loss + Dfake_loss + grad_real
                Dloss.backward()
                optimizerD.step()
                # update G network
                for p in modelD_.parameters():
                    p.requires_grad = False
                optimizer.zero_grad()
                # compute D loss
                out = modelD_(curr_z0, c=y) + modelD_(reflow_z0_rescon, c=y)
                Gloss = F.softplus(-out).mean()
            else:
                optimizer.zero_grad()

            # consistenncy + reflow
            if args.model_ckpt:
                if global_step < args.warm_up_con:
                    con_loss = F.mse_loss(curr_z0, post_z0)
                    loss = con_loss
                    reflow_loss = torch.tensor(0.)
                else:
                    reflow_loss = F.mse_loss(reflow_z0_rescon, reflow_z0.detach()) #+ F.mse_loss(pred_z0, z_0) + F.mse_loss(pred_z0_, z_0)
                    # update inner loop
                    if global_step % args.inner_skip == 0:
                        con_loss = F.mse_loss(curr_z0, post_z0)
                        loss = reflow_loss + con_loss
                    else:
                        loss = reflow_loss
                    # gan loss
                    if args.use_gan:
                        loss += Gloss
            else:
                con_loss = F.mse_loss(curr_z0, post_z0)
                flow_loss = F.mse(curr_z0, z0)
                loss = con_loss + flow_loss
                if args.use_gan:
                    loss += Gloss
            # optimize 
            loss.backward()
            optimizer.step()
            # update ema target
            flow.ema_model.ema_step(args.target_ema_decay, flow.model)

            global_step += 1
            if iteration % 100 == 0:
                if rank == 0:
                    # Measure training speed:
                    end_time = time()
                    steps_per_sec = 100 / (end_time - start_time)
                    if args.use_gan:
                            logger.info('epoch {} iteration{}, Loss: {}, FLoss: {}, CONLoss: {}, Gloss: {}, Dloss {} Train Steps/Sec: {:.2f}'.format(
                                epoch,
                                iteration, 
                                loss.item(), 
                                reflow_loss.item() if args.model_ckpt else flow_loss.item(), 
                                con_loss.item(), 
                                Gloss.item(),
                                Dloss.item(),
                                steps_per_sec))
                    else:
                        logger.info('epoch {} iteration{}, Loss: {}, FLoss: {}, CONLoss: {}, Train Steps/Sec: {:.2f}'.format(
                            epoch,
                            iteration, 
                            loss.item(), 
                            reflow_loss.item() if args.model_ckpt else flow_loss.item(), 
                            con_loss.item(), 
                            steps_per_sec))
                    start_time = time()

        if not args.no_lr_decay:
            scheduler.step()
            if args.use_gan:
                schedulerD.step()

        if rank == 0:
            if epoch % args.plot_every == 0:
                with torch.no_grad():
                    rand = torch.randn_like(z0)[:4]
                    if y is not None:
                        y = y[:4]
                    sample_model = partial(model, y=y)
                    traj, x0_list = flow.sample_ode_generative(rand, args.num_sample_timesteps)
                    fake_image = traj[-1]
                    fake_image = first_stage_model.decode(fake_image / args.scale_factor).sample
                traj = torch.cat(traj, dim=0)
                traj = first_stage_model.decode(traj / args.scale_factor).sample
                x0_list = torch.cat(x0_list, dim=0)
                x0_list = first_stage_model.decode(x0_list / args.scale_factor).sample
                torchvision.utils.save_image(x0_list, os.path.join(exp_path, 'x0_epoch_{}.png'.format(epoch)), normalize=True, nrow=4)
                torchvision.utils.save_image(traj, os.path.join(exp_path, 'traj_epoch_{}.png'.format(epoch)), normalize=True, nrow=4)
                torchvision.utils.save_image(fake_image, os.path.join(exp_path, 'image_epoch_{}.png'.format(epoch)), normalize=True)
                logger.info("Finish sampling")

            if args.save_content:
                if epoch % args.save_content_every == 0:
                    logger.info('Saving content.')
                    if args.use_gan:
                        content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                                'model_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'modelD_dict': modelD.state_dict(), 'optimizerD': optimizerD.state_dict(),
                                'schedulerD': schedulerD.state_dict()}
                    else:
                        content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                                'model_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict()}
                    torch.save(content, os.path.join(exp_path, 'content.pth'))

            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizer.swap_parameters_with_ema(store_params_in_ema=True)
                torch.save(model.state_dict(), os.path.join(exp_path, 'model_{}.pth'.format(epoch)))
                if args.use_ema:
                    optimizer.swap_parameters_with_ema(store_params_in_ema=True)


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

    # discriminator
    parser.add_argument('--lrD', type=float, default=1e-4, help='learning rate d')
    parser.add_argument('--d_base_channels', type=int, default=16384, help='number of discriminator base channels')
    parser.add_argument('--r1_gamma', type=float, default=1., help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None, help='lazy regulariation.')
    parser.add_argument('--d_temb_channels', type=int, default=256, help='number of discriminator temb channels')

    # training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--datadir', default='./data')
    parser.add_argument('--use_fp16', action='store_true', default=False)
    parser.add_argument('--use_grad_checkpointing', action='store_true', default=False, help="Enable gradient checkpointing for mem saving")
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate g')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam')
    parser.add_argument('--no_lr_decay',action='store_true', default=False)
    parser.add_argument('--use_ema', action='store_true', default=False, help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
    parser.add_argument('--target_ema_decay', type=float, default=0.95, help='decay rate for target EMA model')

    # consistency + gan
    parser.add_argument('--init_threshold', type=float, default=0.2, help='training threshold')
    parser.add_argument('--warm_up_con', type=int, default=0, help='warm up consistency loss')
    parser.add_argument('--use_gan', action='store_true', default=False)
    parser.add_argument('--inner_skip', type=int, default=1)

    # saving
    parser.add_argument('--save_content', action='store_true', default=False)
    parser.add_argument('--save_content_every', type=int, default=10, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=25, help='save ckpt every x epochs')
    parser.add_argument('--plot_every', type=int, default=1, help='plot every x epochs')

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

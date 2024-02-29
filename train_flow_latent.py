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
from models import create_network
from EMA import EMA, EMAMODEL

from models.augment import AugmentPipe
from stochastic_flow import stochastic_forward


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def get_weight(model):
    size_all_mb = sum(p.numel() for p in model.parameters()) / 1024**2
    return size_all_mb


def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)


def sample_from_model(model, x_0):
    t = torch.tensor([1., 0.], dtype=x_0.dtype, device="cuda")
    fake_image = odeint(model, x_0, t, atol=1e-5, rtol=1e-5, adjoint_params=model.func.parameters())
    return fake_image


#%%
def train(rank, gpu, args):
    from diffusers.models import AutoencoderKL
    if args.faster_training:
        # faster training
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    device = torch.device('cuda:{}'.format(gpu))
    dtype = torch.float16 if args.use_fp16 else torch.float32

    exp = args.exp
    parent_dir = "./saved_info/latent_flow/{}".format(args.dataset)
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
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{exp_path}/log.txt")]
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
                                               drop_last = True)

    augment_pipe = AugmentPipe(p=args.augment, xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1) if args.augment else None
    if args.augment:
        args.augment_dim = 9

    model = create_network(args).to(device, dtype=dtype)
    if args.use_grad_checkpointing and "DiT" in args.model_type:
        model.set_gradient_checkpointing()
    ema = EMAMODEL(model)

    first_stage_model = AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt).to(device, dtype=dtype)
    first_stage_model = first_stage_model.eval()
    first_stage_model.train = False
    for param in first_stage_model.parameters():
        param.requires_grad = False

    logger.info('AutoKL size: {:.3f}MB'.format(get_weight(first_stage_model)))
    logger.info('FM size: {:.3f}MB'.format(get_weight(model)))

    broadcast_params(model.parameters())

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # if args.use_ema:
    #     optimizer = EMA(optimizer, ema_decay=args.ema_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, eta_min=1e-5)
    # Ensure EMA is initialized with synced weights
    ema.ema_step(decay_rate=0, model=model)  

    #ddp
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=False)

    if args.compile:
        model = torch.compile(model)

    if args.resume or os.path.exists(os.path.join(exp_path, 'content.pth')):
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        model.load_state_dict(checkpoint['model_dict'])
        ema.ema_model.load_state_dict(checkpoint['ema_model_dict'])

        # load G
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        global_step = checkpoint["global_step"]

        logger.info("=> resume checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        del checkpoint

    elif args.model_ckpt and os.path.exists(args.model_ckpt):
        checkpoint_file = args.model_ckpt
        checkpoint = torch.load(checkpoint_file, map_location=device)
        epoch = int(args.model_ckpt.split("_")[-1][:-4])
        init_epoch = 0
        model.load_state_dict(checkpoint['model'])
        ema.ema_model.load_state_dict(checkpoint['ema_model_dict'])
        global_step = 0

        logger.info("=> loaded checkpoint (epoch {})"
                  .format(epoch))
        del checkpoint
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    ema.ema_step(decay_rate=0, model=model.module) # remember to remove

    use_label = True if "imagenet" in args.dataset else False
    is_latent_data = True if "latent" in args.dataset else False
    start_time = time()
    for epoch in range(init_epoch, args.num_epoch+1):
        train_sampler.set_epoch(epoch)

        for iteration, (x, y) in enumerate(data_loader):
            x_0 = x.to(device, dtype=dtype, non_blocking=True)
            y = None if not use_label else y.to(device, non_blocking=True)
            x_0, augment_labels = augment_pipe(x_0) if augment_pipe is not None else (x_0, None)
            model.zero_grad()
            if is_latent_data:
                z_0 = x_0 * args.scale_factor
            else:
                z_0 = first_stage_model.encode(x_0).latent_dist.sample().mul_(args.scale_factor)
            #sample t
            if args.discrete:
                t = torch.randint(1,1001,(z_0.shape[0],)).to(z_0.device).float()/1000
            else:
                t = torch.rand((z_0.size(0),), dtype=dtype, device=device)
                t = t * (1 - 1e-5) + 1e-5
            
            t = t.view(-1, 1, 1, 1)
            z_1 = torch.randn_like(z_0)
            # corrected notation: 1 is real noise, 0 is real data
            # z_t = (1 - t) * z_0 + (1e-5 + (1 - 1e-5) * t) * z_1
            # u = (1 - 1e-5) * z_1 - z_0
            z_t = (1 - t) * z_0 + t * z_1
            u = z_1 - z_0
            z_t, u, t = stochastic_forward(z_t, u, t, form=args.gamma_form)

            v = model(t.squeeze(), z_t, y) # augment_labels)
            loss = F.mse_loss(v, u)
            loss.backward()
            optimizer.step()
            global_step += 1
            if iteration % 100 == 0:
                if rank == 0:
                    end_time = time()
                    steps_per_sec = 100 / (end_time - start_time)
                    logger.info('epoch {} iteration{}, Loss: {}, Train Steps/Sec: {:.2f}'.format(epoch, iteration, loss.item(), steps_per_sec))
                    start_time = time()

            # update ema 
            ema.ema_step(args.ema_decay, model.module)

        if not args.no_lr_decay:
            scheduler.step()

        if rank == 0:
            if epoch % args.plot_every == 0:

                with torch.no_grad():
                    rand = torch.randn_like(z_0)[:4]
                    if y is not None:
                        y = y[:4]
                    sample_model = partial(ema.ema_model, y=y)
                    # sample_func = lambda t, x: model(t, x, y=y)
                    fake_sample = sample_from_model(sample_model, rand)[-1]
                    fake_image = first_stage_model.decode(fake_sample / args.scale_factor).sample
                # torchvision.utils.save_image(fake_sample, os.path.join(exp_path, 'sample_epoch_{}.png'.format(epoch)), normalize=True, value_range=(-1, 1))
                torchvision.utils.save_image(fake_image, os.path.join(exp_path, 'image_epoch_{}.png'.format(epoch)), normalize=True, value_range=(-1, 1))
                logger.info("Finish sampling")

            if args.save_content:
                if epoch % args.save_content_every == 0:
                    logger.info('Saving content.')
                    content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                               'model_dict': model.state_dict() if not args.compile else model._orig_mod.state_dict(), 
                               'optimizer': optimizer.state_dict(),
                               'scheduler': scheduler.state_dict(),
                               'ema_model_dict': ema.ema_model.state_dict(),
                               }

                    torch.save(content, os.path.join(exp_path, 'content.pth'))

            if epoch % args.save_ckpt_every == 0:
                # torch.save(model.state_dict(), os.path.join(exp_path, 'model_{}.pth'.format(epoch)))
                torch.save({
                    "model": model.state_dict() if not args.compile else model._orig_mod.state_dict(), 
                    "ema_model": ema.ema_model.state_dict(),
                }, os.path.join(exp_path, 'model_{}.pth'.format(epoch)))
                # if args.use_ema:
                #     optimizer.swap_parameters_with_ema(store_params_in_ema=True)
                #     torch.save(model.state_dict(), os.path.join(exp_path, 'model_ema{}.pth'.format(epoch)))
                #     optimizer.swap_parameters_with_ema(store_params_in_ema=True)


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
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')
    parser.add_argument('--f', type=int, default=8,
                            help='downsample rate of input image by the autoencoder')
    parser.add_argument('--scale_factor', type=float, default=0.18215,
                            help='size of image')
    parser.add_argument('--num_in_channels', type=int, default=3,
                            help='in channel image')
    parser.add_argument('--num_out_channels', type=int, default=3,
                            help='in channel image')
    parser.add_argument('--nf', type=int, default=256,
                            help='channel of model')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', nargs='+', type=int, default=(16,),
                            help='resolution of applying attention')
    parser.add_argument('--ch_mult', nargs='+', type=int, default=(1,1,2,2,4,4),
                            help='channel mult')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--label_dim', type=int, default=0,
                            help='label dimension, 0 if unconditional')
    parser.add_argument('--augment', type=float, default=0.)
    parser.add_argument('--augment_dim', type=int, default=0,
                            help='dimension of augmented label, 0 if not used')
    parser.add_argument('--num_classes', type=int, default=None,
                            help='num classes')
    parser.add_argument('--label_dropout', type=float, default=0.,
                            help='Dropout probability of class labels for classifier-free guidance')
    parser.add_argument('--weight_decay', type=float, default=0.,
                            help='weight decay')
    parser.add_argument('--gamma_form', type=str, default="none",
                            help='gamma form in stochastic forward', choices=['none', 'log', 'sin'])

    # Original ADM
    parser.add_argument('--layout', action='store_true')
    parser.add_argument('--use_origin_adm', action='store_true')
    parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
    parser.add_argument("--resblock_updown", type=bool, default=False)
    parser.add_argument("--use_new_attention_order", type=bool, default=False)
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument("--resamp_with_conv", type=bool, default=True)
    parser.add_argument('--num_heads', type=int, default=4,
                            help='number of head')
    parser.add_argument('--num_head_upsample', type=int, default=-1,
                            help='number of head upsample')
    parser.add_argument('--num_head_channels', type=int, default=-1,
                            help='number of head channels')

    parser.add_argument('--pretrained_autoencoder_ckpt', type=str, default="../stabilityai/sd-vae-ft-ema")

    # training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--datadir', default='./data')
    parser.add_argument('--num_timesteps', type=int, default=200)
    parser.add_argument('--use_fp16', action='store_true', default=False)
    parser.add_argument('--use_grad_checkpointing', action='store_true', default=False,
        help="Enable gradient checkpointing for mem saving")
    parser.add_argument('--compile', action='store_true', default=False)

    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)

    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate g')

    parser.add_argument('--beta1', type=float, default=0.5,
                            help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                            help='beta2 for adam')
    parser.add_argument('--no_lr_decay',action='store_true', default=False)

    parser.add_argument('--use_ema', action='store_true', default=False,
                            help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
    parser.add_argument('--faster_training',action='store_true', default=False)
    parser.add_argument('--discrete', action='store_true', default=False)


    parser.add_argument('--save_content', action='store_true', default=False)
    parser.add_argument('--save_content_every', type=int, default=10, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=25, help='save ckpt every x epochs')
    parser.add_argument('--plot_every', type=int, default=5, help='plot every x epochs')

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

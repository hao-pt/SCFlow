# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import copy
import logging
import os
import shutil
from functools import partial
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from omegaconf import OmegaConf
from torch import nn, optim
from torch.multiprocessing import Process
from tqdm import tqdm

from datasets_prep import get_dataset
from distill.flows import ConsistencyFlow
from EMA import EMAMODEL
from models import create_discriminator, create_network


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
        t = t.detach().cpu().numpy()
        t = t.squeeze()
        for key in self.register_key:
            self.losses[key] = np.concatenate(
                (self.losses[key], track_loss[key].detach().squeeze().cpu().numpy()),
            )
        self.times = np.concatenate((self.times, t.reshape(-1)))


def batch_mse(input, target):
    return torch.mean((input - target) ** 2, dim=(1, 2, 3))


def huber_loss(x, y, c=0.01):
    c = torch.tensor(c, device=x.device)
    return (torch.sqrt((x - y) ** 2.0 + c**2) - c).mean(dim=(1, 2, 3))


def grad_penalty_call(args, D_real, x_t):
    grad_real = torch.autograd.grad(
        outputs=D_real.sum(), inputs=x_t, create_graph=True,
    )[0]
    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()

    grad_penalty = args.r1_gamma / 2 * grad_penalty
    return grad_penalty


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def get_weight(model):
    size_all_mb = sum(p.numel() for p in model.parameters()) / 1024**2
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


def train(rank, gpu, args):
    from diffusers.models import AutoencoderKL

    if args.faster_training:
        # faster training
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    device = torch.device(f"cuda:{gpu}")
    dtype = torch.float16 if args.use_fp16 else torch.float32

    exp = args.exp
    if args.model_ckpt:
        parent_dir = f"./saved_info/reflow_consistency/{args.dataset}"
    else:
        parent_dir = f"./saved_info/flow_consistency/{args.dataset}"

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
                logging.FileHandler(f"{exp_path}/log.txt"),
            ],
        )
        # Creating an object
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.info(f"Exp path: {exp_path}")

    batch_size = args.batch_size
    dataset = get_dataset(args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=args.world_size, rank=rank,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    model = create_network(args).to(device, dtype=dtype)
    if args.use_grad_checkpointing and "DiT" in args.model_type:
        model.set_gradient_checkpointing()
    ema = EMAMODEL(model)

    first_stage_model = AutoencoderKL.from_pretrained(
        args.pretrained_autoencoder_ckpt,
    ).to(device, dtype=dtype)
    first_stage_model = first_stage_model.eval()
    first_stage_model.train = False
    for param in first_stage_model.parameters():
        param.requires_grad = False
    if args.use_gan:
        modelD = create_discriminator(args).to(device, dtype=dtype)

    if rank == 0:
        logger.info(f"AutoKL size: {get_weight(first_stage_model):.3f}MB")
        logger.info(f"FM size: {get_weight(model):.3f}MB")
        if args.use_gan:
            logger.info(f"Discriminator size: {get_weight(modelD):.3f}MB")

    broadcast_params(model.parameters())
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.num_epoch, eta_min=1e-5,
    )
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[gpu], find_unused_parameters=False,
    )
    model.train()
    if args.use_gan:
        broadcast_params(modelD.parameters())
        optimizerD = optim.AdamW(modelD.parameters(), lr=args.lrD, weight_decay=0.0)
        schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizerD, args.num_epoch, eta_min=1e-5,
        )
        modelD = nn.parallel.DistributedDataParallel(
            modelD, device_ids=[gpu], find_unused_parameters=True,
        )
        modelD.train()
        modelD_ = modelD

    # Ensure EMA is initialized with synced weights
    ema.ema_step(decay_rate=0, model=model)

    if args.model_ckpt:
        ckpt = torch.load(args.model_ckpt, map_location=device)
        model.load_state_dict(ckpt)
        teacher = copy.deepcopy(model)
        teacher.eval()
    else:
        teacher = None

    if args.resume or os.path.exists(os.path.join(exp_path, "content.pth")):
        checkpoint_file = os.path.join(exp_path, "content.pth")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint["epoch"]
        epoch = init_epoch
        model.load_state_dict(checkpoint["model_dict"])
        ema.ema_model.load_state_dict(checkpoint["ema_model_dict"])

        # load G
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = checkpoint["global_step"]
        # load D
        if args.use_gan:
            modelD.load_state_dict(checkpoint["modelD_dict"])
            optimizerD.load_state_dict(checkpoint["optimizerD"])
            schedulerD.load_state_dict(checkpoint["schedulerD"])
        if rank == 0:
            logger.info("=> resume checkpoint (epoch {})".format(checkpoint["epoch"]))
        del checkpoint
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    if args.compile:
        model = torch.compile(model)

    teacher = copy.deepcopy(model)
    teacher.eval()
    target_ema = EMAMODEL(model)
    model_ = model

    flow = ConsistencyFlow(
        device,
        model=model_,
        ema_model=target_ema,
        threshold=args.init_threshold,
        trunc_threshold=args.trunc_threshold,
        pretrained_model=teacher,
    )
    use_label = True if "imagenet" in args.dataset else False
    register_key = ["flow_loss", "con_loss", "reflow_loss"]
    loss_record = LossRecord(register_key=register_key)
    record_path = os.path.join(exp_path, "record")
    if rank == 0:
        if not os.path.exists(record_path):
            os.makedirs(record_path)
    start_time = time()
    for epoch in range(init_epoch, args.num_epoch + 1):
        train_sampler.set_epoch(epoch)
        # reset record
        loss_record.reset()
        for iteration, (x, y) in enumerate(tqdm(data_loader)):
            x0 = x.to(device, dtype=dtype, non_blocking=True)
            y = None if not use_label else y.to(device, non_blocking=True)
            z0 = (
                first_stage_model.encode(x0)
                .latent_dist.sample()
                .mul_(args.scale_factor)
            )
            # set grad for gan
            z0.requires_grad = True
            x0.requires_grad = True
            z1 = torch.randn_like(z0)
            t = torch.rand((z1.shape[0], 1), device=device).float()
            model_kwargs = {}
            (
                curr_vt,
                _,
                _,
                gt_flow,
                curr_z0,
                post_z0,
                prev_z0,
                reflow_z0_rescon,
                reflow_z0,
            ) = flow.get_train_tuple(z0, z1, t=t, model_kwargs=model_kwargs)
            # gan part
            if args.use_gan and global_step > args.warm_up_gan:
                for p in modelD_.parameters():
                    p.requires_grad = True
                optimizerD.zero_grad()
                # calc D real
                Dreal = modelD_(z0, c=y)
                Dreal_loss = F.softplus(-Dreal).mean()
                # penalty discriminator
                grad_real = 0.0
                if args.lazy_reg is None or global_step % args.lazy_reg == 0:
                    grad_real = grad_penalty_call(args, Dreal, z0)
                Dfake = modelD_(reflow_z0.detach().requires_grad_(), c=y)
                Dfake_loss = F.softplus(Dfake).mean()
                Dloss = Dreal_loss + Dfake_loss + grad_real
                Dloss.backward()
                optimizerD.step()
                # update G network
                for p in modelD_.parameters():
                    p.requires_grad = False
                optimizer.zero_grad()
                # compute D loss
                out = modelD_(reflow_z0, c=y)
                Gloss = F.softplus(-out).mean()
            else:
                optimizer.zero_grad()
                Gloss = torch.zeros(1)
                Dloss = torch.zeros(1)
            # setup to record losses
            my_vars = locals()
            loss_dict = {}

            # consistenncy + reflow
            reflow_loss = torch.zeros(args.batch_size)
            if args.model_ckpt:
                con_loss = huber_loss(curr_z0, post_z0)
                loss = con_loss.mean()
                # reflow loss
                if global_step > args.warm_up_reflow:
                    reflow_loss = batch_mse(reflow_z0_rescon, reflow_z0.detach())
                    loss += args.scale_reflow * reflow_loss.mean()
                # gan loss
                if global_step > args.warm_up_gan and args.use_gan:
                    loss += args.scale_gan * Gloss
                # bidirectional flow consistency
                if global_step > args.warm_up_inverse:
                    prev_con = huber_loss(curr_z0, prev_z0)
                    loss += args.scale_inverse * prev_con.mean()
            else:
                con_loss = batch_mse(curr_z0, post_z0)
                loss = con_loss.mean()
                if global_step >= args.warm_up_reflow:
                    # reflow
                    reflow_loss = batch_mse(curr_z0, z0)
                    loss += args.scale_reflow * reflow_loss.mean()
                if global_step >= args.warm_up_gan and args.use_gan:
                    loss += args.scale_gan * Gloss
                if global_step >= args.warm_up_inverse:
                    prev_con = huber_loss(curr_z0, prev_z0)
                    loss += args.scale_inverse * prev_con.mean()
            # record losses
            my_vars = locals()
            for key in register_key:
                try:
                    loss_dict[key] = my_vars[key]
                except KeyError:
                    loss_dict[key] = torch.zeros_like(t)
            loss_record.add(t, loss_dict)

            # optimize
            loss.backward()
            optimizer.step()
            # update ema target
            flow.ema_model.ema_step(args.target_ema_decay, flow.model)
            ema.ema_step(args.ema_decay, flow.model.module)

            if args.progressive_iter:
                if rank == 0:
                    if (
                        (global_step % args.progressive_iter == 0)
                        and (flow.threshold <= 1.0)
                        and global_step > 0
                    ):
                        flow.threshold += 0.05
                        logger.info(f"Increase threshold to  {flow.threshold}")

            global_step += 1
            if iteration % 100 == 0:
                if rank == 0:
                    # Measure training speed:
                    end_time = time()
                    steps_per_sec = 100 / (end_time - start_time)
                    if args.use_gan:
                        logger.info(
                            "epoch {} iteration{}, Loss: {}, FLoss: {}, CONLoss: {}, Gloss: {}, Dloss {} Train Steps/Sec: {:.2f}".format(
                                epoch,
                                iteration,
                                loss.item(),
                                0
                                if global_step <= args.warm_up_reflow
                                else reflow_loss.mean().item(),
                                con_loss.mean().item(),
                                Gloss.item(),
                                Dloss.item(),
                                steps_per_sec,
                            ),
                        )
                    else:
                        logger.info(
                            "epoch {} iteration{}, Loss: {}, FLoss: {}, CONLoss: {}, Train Steps/Sec: {:.2f}".format(
                                epoch,
                                iteration,
                                loss.item(),
                                0
                                if global_step <= args.warm_up_reflow
                                else reflow_loss.mean().item(),
                                con_loss.mean().item(),
                                steps_per_sec,
                            ),
                        )
                    start_time = time()

        if args.lr_decay:
            scheduler.step()
            if args.use_gan:
                schedulerD.step()

        if rank == 0:
            loss_record.plot(os.path.join(record_path, f"Epoch_{epoch}.jpg"))
            if epoch % args.plot_every == 0:
                with torch.no_grad():
                    rand = torch.randn_like(z0)[:4]
                    if y is not None:
                        y = y[:4]
                    sample_model = partial(ema.ema_model, y=y)
                    traj, x0_list = flow.sample_ode_generative(
                        rand, args.num_sample_timesteps,
                    )
                    fake_image = traj[-1]
                    fake_image = first_stage_model.decode(
                        fake_image / args.scale_factor,
                    ).sample
                traj = torch.cat(traj, dim=0)
                traj = first_stage_model.decode(traj / args.scale_factor).sample
                x0_list = torch.cat(x0_list, dim=0)
                x0_list = first_stage_model.decode(x0_list / args.scale_factor).sample
                torchvision.utils.save_image(
                    x0_list,
                    os.path.join(exp_path, f"x0_epoch_{epoch}.png"),
                    normalize=True,
                    nrow=4,
                )
                torchvision.utils.save_image(
                    traj,
                    os.path.join(exp_path, f"traj_epoch_{epoch}.png"),
                    normalize=True,
                    nrow=4,
                )
                torchvision.utils.save_image(
                    fake_image,
                    os.path.join(exp_path, f"image_epoch_{epoch}.png"),
                    normalize=True,
                )
                logger.info("Finish sampling")

            if epoch % args.save_content_every == 0:
                logger.info("Saving content.")
                if args.use_gan:
                    content = {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "args": args,
                        "model_dict": model_.state_dict()
                        if not args.compile
                        else model_._orig_mod.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "modelD_dict": modelD.state_dict(),
                        "optimizerD": optimizerD.state_dict(),
                        "schedulerD": schedulerD.state_dict(),
                        "ema_model_dict": ema.ema_model.state_dict(),
                    }
                else:
                    content = {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "args": args,
                        "model_dict": model_.state_dict()
                        if not args.compile
                        else model_._orig_mod.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "ema_model_dict": ema.ema_model.state_dict(),
                    }
                torch.save(content, os.path.join(exp_path, "content.pth"))

            if epoch % args.save_ckpt_every == 0:
                torch.save(
                    {
                        "model": model_.state_dict()
                        if not args.compile
                        else model_._orig_mod.state_dict(),
                        "ema_model": ema.ema_model.state_dict(),
                    },
                    os.path.join(exp_path, f"model_{epoch}.pth"),
                )


def init_processes(rank, size, fn, args):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = args.master_address
    os.environ["MASTER_PORT"] = args.master_port
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(
        backend="nccl", init_method="env://", rank=rank, world_size=size,
    )
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()


def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ddgan parameters")
    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
        help="seed used for initialization",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default=None,
        help="Model ckpt to init from",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="adm",
        help="model_type",
        choices=[
            "adm",
            "ncsn++",
            "ddpm++",
            "DiT-B/2",
            "DiT-L/2",
            "DiT-L/4",
            "DiT-XL/2",
        ],
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="size of image",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsample rate of input image by the autoencoder",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=0.18215,
        help="size of image",
    )
    parser.add_argument(
        "--num_in_channels",
        type=int,
        default=4,
        help="in channel image",
    )
    parser.add_argument(
        "--num_out_channels",
        type=int,
        default=4,
        help="in channel image",
    )
    parser.add_argument(
        "--nf",
        type=int,
        default=256,
        help="channel of model",
    )
    parser.add_argument(
        "--num_res_blocks",
        type=int,
        default=2,
        help="number of resnet blocks per scale",
    )
    parser.add_argument(
        "--attn_resolutions",
        nargs="+",
        type=int,
        default=(16, 8),
        help="resolution of applying attention",
    )
    parser.add_argument(
        "--ch_mult",
        nargs="+",
        type=int,
        default=(1, 2, 2, 2),
        help="channel mult",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="drop-out rate",
    )
    parser.add_argument(
        "--label_dim",
        type=int,
        default=0,
        help="label dimension, 0 if unconditional",
    )
    parser.add_argument(
        "--augment_dim",
        type=int,
        default=0,
        help="dimension of augmented label, 0 if not used",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="num classes",
    )
    parser.add_argument(
        "--label_dropout",
        type=float,
        default=0.0,
        help="Dropout probability of class labels for classifier-free guidance",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--faster_training",
        action="store_true",
        default=False,
    )

    # Original ADM
    parser.add_argument(
        "--layout",
        action="store_true",
    )
    parser.add_argument(
        "--use_origin_adm",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use_scale_shift_norm",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--resblock_updown",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--use_new_attention_order",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--centered",
        action="store_false",
        default=True,
        help="-1,1 scale",
    )
    parser.add_argument(
        "--resamp_with_conv",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="number of head",
    )
    parser.add_argument(
        "--num_head_upsample",
        type=int,
        default=-1,
        help="number of head upsample",
    )
    parser.add_argument(
        "--num_head_channels",
        type=int,
        default=-1,
        help="number of head channels",
    )
    parser.add_argument(
        "--pretrained_autoencoder_ckpt",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
    )

    # distill
    parser.add_argument(
        "--num_sample_timesteps",
        type=int,
        default=10,
    )

    # discriminator
    parser.add_argument(
        "--lrD",
        type=float,
        default=1e-4,
        help="learning rate d",
    )
    parser.add_argument(
        "--d_base_channels",
        type=int,
        default=16384,
        help="number of discriminator base channels",
    )
    parser.add_argument(
        "--r1_gamma",
        type=float,
        default=1.0,
        help="coef for r1 reg",
    )
    parser.add_argument(
        "--lazy_reg",
        type=int,
        default=None,
        help="lazy regulariation.",
    )
    parser.add_argument(
        "--d_temb_channels",
        type=int,
        default=256,
        help="number of discriminator temb channels",
    )

    # training
    parser.add_argument(
        "--exp",
        default="experiment_cifar_default",
        help="name of experiment",
    )
    parser.add_argument(
        "--dataset",
        default="cifar10",
        help="name of dataset",
    )
    parser.add_argument(
        "--datadir",
        default="./data",
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use_grad_checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing for mem saving",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="input batch size",
    )
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="learning rate g",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.5,
        help="beta1 for adam",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.9,
        help="beta2 for adam",
    )
    parser.add_argument(
        "--lr_decay",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=False,
        help="use EMA or not",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="decay rate for EMA",
    )
    parser.add_argument(
        "--target_ema_decay",
        type=float,
        default=0.95,
        help="decay rate for target EMA model",
    )
    parser.add_argument(
        "--progressive_iter",
        type=int,
        default=None,
        help="training progressive",
    )

    # consistency + gan
    parser.add_argument(
        "--init_threshold",
        type=float,
        default=0.2,
        help="training threshold",
    )
    parser.add_argument(
        "--trunc_threshold",
        type=float,
        default=0.5,
        help="training threshold",
    )
    parser.add_argument(
        "--warm_up_reflow",
        type=int,
        default=0,
        help="warm up consistency loss",
    )
    parser.add_argument(
        "--warm_up_gan",
        type=int,
        default=0,
        help="warm up consistency loss",
    )
    parser.add_argument(
        "--warm_up_inverse",
        type=int,
        default=0,
        help="warm up consistency loss",
    )
    parser.add_argument(
        "--warm_up_flow",
        type=int,
        default=0,
        help="warm up flow matching loss",
    )
    parser.add_argument(
        "--use_gan",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--inner_skip",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--scale_inverse",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--scale_gan",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--scale_reflow",
        type=float,
        default=0.1,
    )

    # saving
    parser.add_argument(
        "--save_content",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save_content_every",
        type=int,
        default=5,
        help="save content for resuming every x epochs",
    )
    parser.add_argument(
        "--save_ckpt_every",
        type=int,
        default=5,
        help="save ckpt every x epochs",
    )
    parser.add_argument(
        "--plot_every",
        type=int,
        default=1,
        help="plot every x epochs",
    )

    ###ddp
    parser.add_argument(
        "--num_proc_node",
        type=int,
        default=1,
        help="The number of nodes in multi node env.",
    )
    parser.add_argument(
        "--num_process_per_node",
        type=int,
        default=1,
        help="number of gpus",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="The index of node.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="rank of process in the node",
    )
    parser.add_argument(
        "--master_address", type=str, default="127.0.0.1", help="address for master",
    )
    parser.add_argument(
        "--master_port", type=str, default="6000", help="port for master",
    )

    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        try:
            torch.multiprocessing.set_start_method("spawn")
        except RuntimeError:
            pass

        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print(
                "Node rank %d, local proc %d, global proc %d"
                % (args.node_rank, rank, global_rank),
            )
            p = Process(
                target=init_processes, args=(global_rank, global_size, train, args),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print("starting in debug mode")

        init_processes(0, size, train, args)

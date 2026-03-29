# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import argparse
import logging
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import shutil
from pathlib import Path

import accelerate
import datasets
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from huggingface_hub import create_repo
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig

from models.discriminator import Discriminator
from sd_distill.flows import ConsistencyFlow
from sd_distill.latent_dataset import LatentText2ImageDataset
from sd_distill.loss import grad_penalty_call, huber_loss
from sd_distill.model_utils import EMAMODEL
from sd_distill.schedulers import ReflowScheduler
from sd_distill.sd_discriminator import SDDiscriminator, copy_weight_from_teacher

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.25.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def gather_mean(accelerator, args, loss):
    # Gather the losses across all processes for logging (if we use distributed training).
    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
    train_loss = avg_loss.item() / args.gradient_accumulation_steps
    return train_loss


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str,
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    raise ValueError(f"{model_class} is not supported.")


# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
def encode_prompt(prompts, text_encoder, tokenizer, is_train=True):
    captions = []
    for caption in prompts:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
        )[0]

    return {"prompt_embeds": prompt_embeds.cpu()}


@torch.no_grad()
def inference(
    args,
    unet,
    vae,
    flow,
    encoded_embeds,
    null_embed,
    generator,
    device,
    weight_dtype,
    is_teacher,
    guidance_scale,
):
    input_shape = (1, 4, args.resolution // 8, args.resolution // 8)
    input_noise = torch.randn(
        input_shape, generator=generator, device=device, dtype=weight_dtype,
    )

    prompt_embed = encoded_embeds["prompt_embeds"]
    prompt_embed = prompt_embed.to(device, weight_dtype)
    if null_embed is not None:
        null_embed = null_embed["prompt_embeds"].to(device, weight_dtype)
    else:
        null_embed = None

    # traj, x0_list = flow.sample_ode_generative(input_noise, args.num_sample_timesteps, encoder_hidden_states=prompt_embed)
    traj, x0_list = flow.ddim_sample_loop_skip(
        args,
        unet,
        noise=input_noise,
        weight_dtype=weight_dtype,
        prompt_embed=prompt_embed,
        prompt_null_embed=null_embed,
        progress=False,
        eta=0.0,
        guidance_scale=guidance_scale,
        skip=None if not is_teacher else 40,
    )  # default: 25-steps for teacher
    pred_original_sample = traj[-1]
    pred_original_sample = pred_original_sample / vae.config.scaling_factor

    image = vae.decode(pred_original_sample.to(dtype=vae.dtype)).sample.float()
    image = (image[0].detach().cpu() * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    x0_seq = torch.cat(
        [vae.decode(x / vae.config.scaling_factor).sample.float() for x in x0_list],
        dim=0,
    )
    x0_seq = (x0_seq.detach().cpu() * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    x0_seq = rearrange(x0_seq, "b c h w -> c h (b w)")

    return image, x0_seq


@torch.no_grad()
def validation(
    accelerator,
    args,
    validation_dicts,
    null_dict,
    unet,
    vae,
    flow,
    weight_dtype,
    is_teacher=False,
    guidance_scale=4.5,
):
    # run inference
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
    with torch.cuda.amp.autocast():
        images, x0_seq = {}, {}
        for prompt, validation_dict in zip(args.validation_prompts, validation_dicts):
            outputs = [
                inference(
                    args,
                    unet,
                    vae,
                    flow,
                    validation_dict,
                    null_dict,
                    generator=generator,
                    device=accelerator.device,
                    weight_dtype=weight_dtype,
                    is_teacher=is_teacher,
                    guidance_scale=guidance_scale,
                )
                for _ in range(args.num_validation_images)
            ]
            images[prompt], x0_seq[prompt] = zip(*outputs)

    return images, x0_seq


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="non-encoded",
        help=("non-encoded or encoded",),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X steps. The validation process consists of running the prompts"
            " `args.validation_prompts` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use EMA model.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="The classifier-free guidance scale.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--noise_offset",
        type=float,
        default=0,
        help="The scale of noise offset.",
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    # reflow
    parser.add_argument(
        "--learning_rate_disc",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--target_ema_unet_decay",
        type=float,
        default=0.95,
        help="EMA decay for target ema unet in reflow",
    )
    parser.add_argument(
        "--time_init_threshold",
        type=float,
        default=20,
        help="time init threshold in reflow",
    )
    parser.add_argument(
        "--trunc_threshold",
        type=float,
        default=20,
        help="truncated threshold of estimated flow (Song technique) in reflow",
    )
    parser.add_argument(
        "--lazy_reg",
        type=int,
        default=None,
        help="lazy regulariation.",
    )
    parser.add_argument(
        "--use_gan",
        action="store_true",
        help="enable gan flag",
    )
    parser.add_argument(
        "--warmup_reflow",
        type=int,
        default=0,
        help="warm up reflow loss",
    )
    parser.add_argument(
        "--warmup_gan",
        type=int,
        default=0,
        help="warm up gan loss",
    )
    parser.add_argument(
        "--warmup_inverse",
        type=int,
        default=0,
        help="warm up inverse loss",
    )
    parser.add_argument(
        "--gan_lamb",
        type=float,
        default=0.1,
        help="gan lambda",
    )
    parser.add_argument(
        "--inv_lamb",
        type=float,
        default=0.1,
        help="inverse lambda",
    )
    parser.add_argument(
        "--reflow_lamb",
        type=float,
        default=0.1,
        help="reflow lambda",
    )
    parser.add_argument(
        "--num_sample_timesteps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--r1_gamma",
        type=float,
        default=1.0,
        help="coef for r1 reg",
    )
    parser.add_argument(
        "--random_init_unet", action="store_true", help="Random initialization for unet",
    )
    parser.add_argument(
        "--plot_teacher_samples", action="store_true", help="Plot teacher samples",
    )
    parser.add_argument(
        "--timestep_scaling_factor",
        type=float,
        default=10.0,
        help=(
            "The multiplicative timestep scaling factor used when calculating the boundary scalings for LCM. The"
            " higher the scaling is, the lower the approximation error, but the default value of 10.0 should typically"
            " suffice."
        ),
    )
    parser.add_argument(
        "--post_conditioning_outputs",
        action="store_true",
        help=(" Enable post conditioning for outputs as LCD model"),
    )
    parser.add_argument(
        "--student_guidance",
        action="store_true",
        help=(" Enable classifier-guidance scale for student outputs"),
    )
    parser.add_argument(
        "--use_sd_discriminator",
        action="store_true",
        help=(" Use SD UNetEncoder for discriminator"),
    )
    parser.add_argument(
        "--use_t_sample_for_gan",
        action="store_true",
        help=(" Use t sample for gan"),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def t_sample_for_gan(batch_size, device):
    # Selected timesteps
    selected_timesteps = [10, 250, 500, 750]
    prob = torch.tensor([0.25, 0.25, 0.25, 0.25])

    # Sample the timesteps
    idx = prob.multinomial(batch_size, replacement=True).to(device)
    timesteps = torch.tensor(selected_timesteps, device=device, dtype=torch.long)[idx]

    return timesteps


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.",
            )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            )
    # Load scheduler, tokenizer and models.
    # noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler = ReflowScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler",
    )

    # DDPMScheduler calculates the alpha and sigma noise schedules (based on the alpha bars) for us
    # Also move the alpha and sigma noise schedules to accelerator.device.
    # alpha_schedule = alpha_schedule.to(accelerator.device)
    # sigma_schedule = sigma_schedule.to(accelerator.device)
    # alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    # sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision,
    )
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    if args.random_init_unet:
        config = UNet2DConditionModel.load_config(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.revision,
            variant=args.variant,
        )
        unet = UNet2DConditionModel.from_config(config)
    else:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.revision,
            variant=args.variant,
        )
    teacher = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
    )
    # freeze parameters of models to save more memory
    # unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    teacher.requires_grad_(False)

    # eval mode
    teacher.eval()
    vae.eval()

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)  # torch.float32
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    teacher.to(accelerator.device, dtype=weight_dtype)

    if args.use_gan:
        # discriminator = UNet2DConditionModel.from_pretrained(
        #     args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        # )
        if args.use_sd_discriminator:
            discriminator = SDDiscriminator.from_config(teacher.config)
            discriminator = copy_weight_from_teacher(discriminator, teacher)
            discriminator.to(accelerator.device, dtype=weight_dtype)
        else:
            discriminator = Discriminator(
                c_dim=0,
                img_resolution=args.resolution // 8,
                img_channels=4,
                channel_base=32768,
                temb_dim=None,
            )
            discriminator.to(accelerator.device, dtype=torch.float)  # NOTE: use fp32
    else:
        discriminator = None

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision,
        )
        ema_unet = EMAModel(
            ema_unet.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=ema_unet.config,
        )

    # target_ema_unet = UNet2DConditionModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    # )
    # target_ema_unet = EMAModel(target_ema_unet.parameters(), decay=args.target_ema_unet_decay, model_cls=UNet2DConditionModel, model_config=target_ema_unet.config)
    target_ema_unet = EMAMODEL(unet)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details.",
                )
            unet.enable_xformers_memory_efficient_attention()
            # discriminator.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly",
            )

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
                # if args.use_gan:
                #     accelerator.save(accelerator.unwrap_model(discriminator).state_dict(), os.path.join(output_dir, "discriminator.pth"))
                for i, model in enumerate(models):
                    if isinstance(model, type(accelerator.unwrap_model(unet))):
                        subfolder = "unet"
                        model.save_pretrained(os.path.join(output_dir, subfolder))
                    elif isinstance(
                        model, type(accelerator.unwrap_model(discriminator)),
                    ):
                        subfolder = "discriminator"
                        if args.use_sd_discriminator:
                            model.save_pretrained(os.path.join(output_dir, subfolder))
                        else:
                            torch.save(
                                model.state_dict(),
                                os.path.join(output_dir, f"{subfolder}.pth"),
                            )

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                print(os.path.join(input_dir, "unet_ema"))
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DConditionModel,
                )
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            # if args.use_gan:
            #     load_model = torch.load(f"{input_dir}/discriminator.pth") #TODO: double check path
            #     discriminator.load_state_dict(load_model)

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    load_model = UNet2DConditionModel.from_pretrained(
                        input_dir, subfolder="unet",
                    )
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                elif isinstance(model, type(accelerator.unwrap_model(discriminator))):
                    if args.use_sd_discriminator:
                        load_model = SDDiscriminator.from_pretrained(
                            input_dir, subfolder="discriminator",
                        )
                        model.register_to_config(**load_model.config)
                        model.load_state_dict(load_model.state_dict())
                    else:
                        ## StyleGAN discriminator
                        load_model = torch.load(
                            f"{input_dir}/discriminator.pth",
                        )  # TODO: double check path
                        model.load_state_dict(load_model)

                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if discriminator is not None and args.use_sd_discriminator:
            discriminator.enable_gradient_checkpointing()
        # teacher.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )
        args.learning_rate_disc = (
            args.learning_rate_disc
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`",
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
        optimizer_disc_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            data_dir=args.train_data_dir,
        )
    else:
        # data_files = {}
        # if args.train_data_dir is not None:
        #     data_files["train"] = os.path.join(args.train_data_dir, "**")
        # dataset = load_dataset(
        #     "imagefolder",
        #     data_files=data_files,
        #     cache_dir=args.cache_dir,
        # )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

        dataset = LatentText2ImageDataset(args.train_data_dir)

    if args.dataset_type != "encoded":
        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = dataset["train"].column_names

        # 6. Get the column names for input/target.
        dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name)
        if args.image_column is None:
            image_column = (
                dataset_columns[0] if dataset_columns is not None else column_names[0]
            )
        else:
            image_column = args.image_column
            if image_column not in column_names:
                raise ValueError(
                    f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}",
                )
        if args.caption_column is None:
            caption_column = (
                dataset_columns[1] if dataset_columns is not None else column_names[1]
            )
        else:
            caption_column = args.caption_column
            if caption_column not in column_names:
                raise ValueError(
                    f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}",
                )

        # Preprocessing the datasets.
        # We need to tokenize input captions and transform the images.
        def tokenize_captions(examples, is_train=True):
            captions = []
            for caption in examples[caption_column]:
                if isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    # take a random caption if there are multiple
                    captions.append(random.choice(caption) if is_train else caption[0])
                else:
                    raise ValueError(
                        f"Caption column `{caption_column}` should contain either strings or lists of strings.",
                    )
            inputs = tokenizer(
                captions,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return inputs.input_ids

        # Preprocessing the datasets.
        train_transforms = transforms.Compose(
            [
                transforms.Resize(
                    args.resolution, interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(args.resolution)
                if args.center_crop
                else transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip()
                if args.random_flip
                else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ],
        )

        def preprocess_train(examples):
            images = [image.convert("RGB") for image in examples[image_column]]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            examples["input_ids"] = tokenize_captions(examples)
            return examples

        def collate_fn(examples):
            pixel_values = torch.stack(
                [example["pixel_values"] for example in examples],
            )
            pixel_values = pixel_values.to(
                memory_format=torch.contiguous_format,
            ).float()
            input_ids = torch.stack([example["input_ids"] for example in examples])
            return {"pixel_values": pixel_values, "input_ids": input_ids}
    else:

        def collate_fn(examples):
            latent_values = torch.stack([example[0] for example in examples])
            latent_values = latent_values.to(
                memory_format=torch.contiguous_format,
            ).float()
            prompt_embeds = torch.stack([example[1] for example in examples])
            return {"latent_values": latent_values, "prompt_embeds": prompt_embeds}

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"]
                .shuffle(seed=args.seed)
                .select(range(args.max_train_samples))
            )
        # Set the training transforms
        train_dataset = (
            dataset
            if args.dataset_type == "encoded"
            else dataset["train"].with_transform(preprocess_train)
        )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps,
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    if discriminator is not None:
        optimizer_disc = optimizer_disc_cls(
            discriminator.parameters(),
            lr=args.learning_rate_disc,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        lr_scheduler_disc = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer_disc,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
        )

        # Prepare everything with our `accelerator`.
        (
            unet,
            discriminator,
            optimizer,
            optimizer_disc,
            train_dataloader,
            lr_scheduler,
            lr_scheduler_disc,
        ) = accelerator.prepare(
            unet,
            discriminator,
            optimizer,
            optimizer_disc,
            train_dataloader,
            lr_scheduler,
            lr_scheduler_disc,
        )
    else:
        lr_scheduler_disc = None
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler,
        )

    if args.use_ema:
        ema_unet.to(accelerator.device)
    # target_ema_unet = target_ema_unet.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps,
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers("reflow", config=tracker_config)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}",
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.",
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    flow = ConsistencyFlow(
        accelerator.device,
        model=unet,
        ema_model=target_ema_unet,
        threshold=args.time_init_threshold,
        trunc_threshold=args.trunc_threshold,
        pretrained_model=teacher,
        noise_scheduler=noise_scheduler,
        post_conditioning_outputs=args.post_conditioning_outputs,
        student_guidance=args.student_guidance,
    )
    null_dict = encode_prompt([""], text_encoder, tokenizer)
    validation_dicts = [
        encode_prompt([prompt], text_encoder, tokenizer)
        for prompt in args.validation_prompts
    ]
    # plot teacher images
    if accelerator.is_main_process and args.plot_teacher_samples:
        logger.info(
            f"Running Teacher inference... \n Generating {args.num_validation_images} images with prompt:"
            f" {args.validation_prompts}.",
        )
        out_images, _ = validation(
            accelerator,
            args,
            validation_dicts,
            null_dict,
            teacher,
            vae,
            flow,
            weight_dtype,
            is_teacher=True,
            guidance_scale=args.guidance_scale,
        )

        for tracker in accelerator.trackers:
            for prompt in args.validation_prompts:
                for i, image in enumerate(out_images[prompt]):
                    tracker.writer.add_images(
                        f"teacher/{prompt}/{i}",
                        np.asarray(image),
                        global_step,
                        dataformats="CHW",
                    )
        torch.cuda.empty_cache()

    if args.dataset_type == "encoded":
        del text_encoder, tokenizer, vae
        torch.cuda.empty_cache()

    # loss_record = LossRecord(register_key=["Gloss", "Dloss", "con_loss", "reflow_loss"])
    record_path = os.path.join(logging_dir, "record")
    if accelerator.is_main_process:
        os.makedirs(record_path, exist_ok=True)

    # with torch.backends.cuda.sdp_kernel(
    #     enable_flash=True, enable_math=True, enable_mem_efficient=True,
    # ):
    logger.info("Start training ...")
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        train_loss_disc = 0.0
        train_loss_gan = 0.0
        train_loss_con = 0.0
        train_loss_prev_con = 0.0
        train_loss_reflow = 0.0
        # loss_record.reset()
        for step, batch in enumerate(train_dataloader):
            with accelerator.autocast():
                with accelerator.accumulate(unet):
                    if args.dataset_type == "non-encoded":
                        # Convert images to latent space
                        latents = vae.encode(
                            batch["pixel_values"].to(
                                accelerator.device, dtype=weight_dtype,
                            ),
                        ).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor

                        # Get the text embedding for conditioning
                        prompt_embeds = text_encoder(batch["input_ids"])[0].to(
                            dtype=weight_dtype,
                        )
                        prompt_null_embeds = (
                            null_dict["prompt_embeds"]
                            .repeat(latents.size(0), 1, 1)
                            .to(accelerator.device, dtype=weight_dtype)
                        )
                    else:
                        latents = batch["latent_values"].to(
                            dtype=weight_dtype, device=accelerator.device,
                        )
                        prompt_embeds = batch["prompt_embeds"].to(
                            dtype=weight_dtype, device=accelerator.device,
                        )
                        prompt_null_embeds = (
                            train_dataset.get_prompt_null_embed()
                            .repeat(latents.size(0), 1, 1)
                            .to(accelerator.device, dtype=weight_dtype)
                        )

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    if args.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1),
                            device=latents.device,
                        )

                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    ).long()
                    # timesteps_cont = (timesteps+1).to(dtype=weight_dtype) / noise_scheduler.config.num_train_timesteps
                    # timesteps = torch.rand((bsz,), device=latents.device).float()

                    model_kwargs = {
                        "prompt_embeds": prompt_embeds,
                        "prompt_null_embeds": prompt_null_embeds,
                    }
                    curr_z0, post_z0, prev_z0, reflow_z0_rescon, reflow_z0, post_t = (
                        flow.get_train_tuple(
                            latents,
                            noise,
                            t=timesteps,
                            guidance_scale=args.guidance_scale,
                            model_kwargs=model_kwargs,
                        )
                    )
                    loss = 0.0
                    if args.use_gan and global_step >= args.warmup_gan:
                        # update disc
                        for p in discriminator.parameters():
                            p.requires_grad = True
                        optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                        if not args.use_sd_discriminator:
                            # set grad for r1 reg of GAN
                            latents.requires_grad = True
                            # calc D real
                            Dreal = discriminator(
                                latents.float(),
                                c=None,
                                force_fp32=True,
                                # timestep=torch.zeros_like(timesteps),
                                # encoder_hidden_states=prompt_embeds
                            )  # .sample
                            Dreal_loss = F.softplus(-Dreal)

                            # penalty discriminator
                            grad_real = 0.0
                            if args.lazy_reg is None or global_step % args.lazy_reg == 0:
                                grad_real = grad_penalty_call(args, Dreal, latents)

                            Dfake = discriminator(
                                reflow_z0.detach().requires_grad_().float(),
                                c=None,
                                force_fp32=True,
                                # timestep=torch.zeros_like(timesteps),
                                # encoder_hidden_states=prompt_embeds
                            )  # .sample
                            Dfake_loss = F.softplus(Dfake)
                            Dloss = (
                                Dreal_loss + Dfake_loss + grad_real.reshape(-1, 1, 1, 1)
                            )
                        else:
                            noise_comma = torch.randn_like(latents)
                            if args.use_t_sample_for_gan:
                                t_comma = t_sample_for_gan(
                                    latents.shape[0], accelerator.device,
                                )
                            else:
                                # t_comma = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                                # old: use post_t
                                t_comma = post_t
                            noisy_latents = flow._noise_forward(
                                latents, noise_comma, t_comma,
                            )
                            noisy_latents.requires_grad = True
                            Dreal = discriminator(
                                noisy_latents,
                                t_comma,
                                encoder_hidden_states=prompt_embeds,
                            )
                            Dreal_loss = F.softplus(-Dreal)

                            noisy_reflow_z0 = flow._noise_forward(
                                reflow_z0.detach(), noise_comma, t_comma,
                            )
                            noisy_reflow_z0.requires_grad = True
                            Dfake = discriminator(
                                noisy_reflow_z0,
                                t_comma,
                                encoder_hidden_states=prompt_embeds,
                            )
                            Dfake_loss = F.softplus(Dfake)
                            Dloss = Dreal_loss + Dfake_loss
                        accelerator.backward(Dloss.mean(), retain_graph=True)
                        optimizer_disc.step()

                        # update G network
                        for p in discriminator.parameters():
                            p.requires_grad = False
                        optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                        if not args.use_sd_discriminator:
                            # compute D loss
                            out = discriminator(
                                reflow_z0.requires_grad_().float(),
                                c=None,
                                force_fp32=True,
                                # timestep=torch.zeros_like(timesteps),
                                # encoder_hidden_states=prompt_embeds
                            )  # .sample
                            Gloss = F.softplus(-out)
                        else:
                            if args.use_t_sample_for_gan:
                                t_comma = t_sample_for_gan(
                                    latents.shape[0], accelerator.device,
                                )
                            else:
                                # t_comma = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                                t_comma = post_t
                            noisy_gen_z0 = flow._noise_forward(
                                reflow_z0.requires_grad_(),
                                torch.randn_like(latents),
                                t_comma,
                            )
                            out = discriminator(
                                noisy_gen_z0,
                                t_comma,
                                encoder_hidden_states=prompt_embeds,
                            )
                            Gloss = F.softplus(-out)
                        loss += args.gan_lamb * Gloss.mean()
                    else:
                        Gloss = torch.zeros(bsz, dtype=weight_dtype)
                        Dloss = torch.zeros(bsz, dtype=weight_dtype)
                        optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                    # consistency + reflow
                    con_loss = huber_loss(curr_z0, post_z0).mean(dim=[1, 2, 3])
                    loss = loss + con_loss.mean()

                    reflow_loss = torch.zeros(bsz, dtype=weight_dtype)
                    if global_step >= args.warmup_reflow and args.reflow_lamb > 0:
                        # reflow_loss = F.mse_loss(reflow_z0, reflow_z0_rescon.detach(), reduction="none").mean(dim=[1,2,3])
                        reflow_loss = F.mse_loss(
                            reflow_z0_rescon, reflow_z0.detach(), reduction="none",
                        ).mean(dim=[1, 2, 3])
                        # reflow_loss = F.mse_loss(reflow_z0_rescon, noise, reduction="none").mean(dim=[1,2,3])
                        loss = loss + args.reflow_lamb * reflow_loss.mean()

                    # bidirectional consistency loss
                    prev_con = torch.zeros(bsz, dtype=weight_dtype)
                    if global_step >= args.warmup_inverse and args.inv_lamb > 0:
                        prev_con = huber_loss(curr_z0, prev_z0)
                        loss = loss + args.inv_lamb * prev_con.mean()
                    # logger.info("Finished loss")

                # logger.info("Gathered loss start")
                # Gather the losses across all processes for logging (if we use distributed training).
                # if accelerator.is_main_process:
                train_loss += gather_mean(accelerator, args, loss)
                if global_step >= args.warmup_reflow and args.reflow_lamb > 0:
                    train_loss_reflow += gather_mean(
                        accelerator, args, reflow_loss.mean(),
                    )
                train_loss_con += gather_mean(accelerator, args, con_loss.mean())
                if global_step >= args.warmup_inverse and args.inv_lamb > 0:
                    train_loss_prev_con += gather_mean(
                        accelerator, args, prev_con.mean(),
                    )
                if (
                    args.use_gan
                    and global_step >= args.warmup_gan
                    and args.gan_lamb > 0
                ):
                    train_loss_disc += gather_mean(accelerator, args, Dloss.mean())
                    train_loss_gan += gather_mean(accelerator, args, Gloss.mean())
                # logger.info("Gathered loss")

                # # setup to record losses
                # my_vars = locals()
                # loss_dict = {}
                # for key in loss_record.register_key:
                #     try:
                #         loss_dict[key] = my_vars[key]
                #     except KeyError:
                #         loss_dict[key] = torch.zeros_like(timesteps)
                # loss_record.add(timesteps, loss_dict)

                # Backpropagate
                accelerator.backward(loss)
                # logger.info("Backward loss")

                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                if args.use_gan:
                    lr_scheduler_disc.step()
                # update ema target
                # flow.ema_model.step(flow.model.parameters())
                flow.ema_model.ema_step(args.target_ema_unet_decay, flow.model)
                # logger.info("Finished training iter")

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                accelerator.log({"train_loss_disc": train_loss_disc}, step=global_step)
                accelerator.log({"train_loss_gan": train_loss_gan}, step=global_step)
                accelerator.log({"train_loss_con": train_loss_con}, step=global_step)
                accelerator.log(
                    {"train_loss_reflow": train_loss_reflow}, step=global_step,
                )
                accelerator.log(
                    {"train_loss_prev_con": train_loss_prev_con}, step=global_step,
                )

                train_loss = 0.0
                if args.use_gan:
                    train_loss_disc = 0.0
                    train_loss_gan = 0.0
                train_loss_con = 0.0
                train_loss_prev_con = 0.0
                train_loss_reflow = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1]),
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints",
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}",
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint,
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}",
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                        # unwrapped_unet = accelerator.unwrap_model(unet)
                        # unet_lora_state_dict = convert_state_dict_to_diffusers(
                        #     get_peft_model_state_dict(unwrapped_unet)
                        # )

                        # StableDiffusionPipeline.save_lora_weights(
                        #     save_directory=save_path,
                        #     unet_lora_layers=unet_lora_state_dict,
                        #     safe_serialization=True,
                        # )

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if (
                global_step % args.validation_steps == 0
                or global_step == args.max_train_steps
            ):
                if (
                    args.validation_prompts is not None
                    and args.num_validation_images > 0
                ):
                    if args.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())

                    vae = AutoencoderKL.from_pretrained(
                        args.pretrained_model_name_or_path,
                        subfolder="vae",
                        revision=args.revision,
                        variant=args.variant,
                    ).to(accelerator.device, dtype=weight_dtype)
                    vae.eval()

                    logger.info(
                        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                        f" {args.validation_prompts}.",
                    )

                    out_images, x0_seq = validation(
                        accelerator,
                        args,
                        validation_dicts,
                        null_dict,
                        unet,
                        vae,
                        flow,
                        weight_dtype,
                        guidance_scale=args.guidance_scale,
                    )

                    for tracker in accelerator.trackers:
                        for prompt in args.validation_prompts:
                            for i, (image, x0_image) in enumerate(
                                zip(out_images[prompt], x0_seq[prompt]),
                            ):
                                tracker.writer.add_images(
                                    f"{prompt}/{i}",
                                    np.asarray(image),
                                    global_step,
                                    dataformats="CHW",
                                )
                                tracker.writer.add_images(
                                    f"x0seq_{prompt}/{i}",
                                    np.asarray(x0_image),
                                    global_step,
                                    dataformats="CHW",
                                )

                    if args.use_ema:
                        # Switch back to the original UNet parameters.
                        ema_unet.restore(unet.parameters())

                    del vae
                    torch.cuda.empty_cache()

                # loss_record.plot(os.path.join(record_path, "Loss_plot_step_{}.jpg".format(global_step)))

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()

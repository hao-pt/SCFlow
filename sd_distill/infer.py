import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import torch
import torch.distributed as dist
import typer
from diffusers import (
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    UNet2DConditionModel,
)
from tqdm import tqdm

from sd_distill.pipeline_rf import RectifiedFlowPipeline

app = typer.Typer()


@app.command()
def main(
    guidance: float = typer.Option(5.0, help="guidance scale"),
    batch_size: int = typer.Option(4, help="Batch_size"),
    step: int = typer.Option(25, help="denoising step"),
    num_samples: int = typer.Option(30000, help="#samples"),
    solver: str = typer.Option("ddpm", help="scheduler"),
    global_seed: int = typer.Option(0, help="#samples"),
    pretrained_unet_ckpt: str = typer.Option(None, help="pretrained unet ckpt"),
    prompts: str = typer.Option(
        "A hyper-realistic photo of a cute cat.",
        help="use either coco2017-5k or coco2014-30k prompts, or a path to a custom prompts JSON file",
    ),
):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    use_coco_prompts = False
    if prompts == "coco2014-30k":
        prompt_path = "sd_distill/coco2014_prompts.json"
        num_samples = 30_000
        use_coco_prompts = True
    elif prompts == "coco2017-5k":
        prompt_path = "sd_distill/coco2017_prompts.json"
        num_samples = 5_000
        use_coco_prompts = True
    elif os.path.isfile(prompts):
        prompt_path = prompts
        num_samples = batch_size
    else:
        prompt_path = None
        num_samples = batch_size

    global_batch_size = batch_size * dist.get_world_size()
    total_samples = int(math.ceil(num_samples / global_batch_size) * global_batch_size)
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    iterations = int(samples_needed_this_gpu // batch_size)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar

    model_id = "pretrained/InstaFlow"  # "XCLIU/2_rectified_flow_from_sd_1_5"
    pipe = RectifiedFlowPipeline.from_pretrained(
        model_id,
        use_safetensors=True,
    )
    if pretrained_unet_ckpt is not None:
        pipe.unet = UNet2DConditionModel.from_pretrained(
            pretrained_unet_ckpt, subfolder="unet_ema", use_safetensors=True,
        )
        print("Loaded pretrained unet checkpoints!!!")
    pipe = pipe.to(device, dtype=torch.bfloat16)
    pipe.unet.config.addition_embed_type = None

    if solver == "dpmsolver++":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif solver == "dpmsolver":
        pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config)

    def disabled_safety_checker(images, clip_input):
        if len(images.shape) == 4:
            num_images = images.shape[0]
            return images, [False] * num_images
        return images, False

    pipe.safety_checker = disabled_safety_checker

    if prompt_path is not None:
        with open(prompt_path) as f:
            data = json.load(f)
        total_images = len(data["labels"][:num_samples])
    else:
        data = None
        total_images = num_samples
    dataname = prompts

    if pretrained_unet_ckpt is None:
        save_dir = f"fid_data/2-rectflow-guidance{guidance}-{step}-{solver}-{total_images}data-{dataname}"
    else:
        ckpt_info, ckpt_name = (
            pretrained_unet_ckpt.rstrip(r"\/").split("/")[-2],
            pretrained_unet_ckpt.rstrip(r"\/").split("/")[-1],
        )
        save_dir = f"fid_data/ours-{ckpt_info}-{ckpt_name}-guidance{guidance}-{step}-{solver}-{total_images}data-{dataname}"

    if not use_coco_prompts:
        save_dir = f"gen_data/ours-{ckpt_info}-{ckpt_name}-guidance{guidance}-{step}-{solver}-{total_images}data-{dataname}"

    total = 0
    for i in pbar:
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        if data is not None:
            paths, batch_prompts = [], []
            for path, prompt in data["labels"][start_idx:end_idx]:
                paths.append(path)
                batch_prompts.append(prompt)
        else:
            batch_prompts = [prompts] * batch_size
            paths = [""] * batch_size

        with torch.no_grad():
            images = pipe(
                batch_prompts,
                num_inference_steps=step,
                guidance_scale=guidance,
            ).images

        for j, (path, image) in enumerate(zip(paths, images)):
            index = j * dist.get_world_size() + rank + total
            os.makedirs(
                os.path.join(
                    save_dir,
                    path.split("/")[0],
                ),
                exist_ok=True,
            )
            image.save(os.path.join(save_dir, path.split("/")[0], f"{index:06d}.png"))

        total += global_batch_size
        dist.barrier()

    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    app()

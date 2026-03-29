import argparse
import functools
import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms.functional as TF
from diffusers.models import AutoencoderKL
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel


def process_prompt_data(save_dir, batch_start, j, image, text):
    np.save(
        f"{save_dir}/{str(batch_start + j).zfill(12)}.npy",
        {"image": image, "text": text},
    )


def wrapper_process_prompt_data(args):
    return process_prompt_data(*args)


class Text2ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        resolution: int = 512,
    ):
        self.path = path
        self.resolution = resolution
        df = pd.read_csv(os.path.join(path, "metadata.csv"))
        self.image_paths = df["file_name"]
        self.text = df["text"]
        assert self.image_paths.shape[0] == self.text.shape[0]

    def transform(self, image):
        # resize image
        image = TF.resize(
            image,
            self.resolution,
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        # get center crop
        image = TF.center_crop(image, output_size=[self.resolution, self.resolution])
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])
        return image

    def __len__(self):
        return self.text.shape[0]

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.image_paths[idx])
        text_prompt = self.text[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, text_prompt


class LaionDagsHubDataset(torch.utils.data.Dataset):
    REPO_URL = "https://dagshub.com/DagsHub-Datasets/LAION-Aesthetics-V2-6.5plus"

    def __init__(
        self, num_samples=500_000, resolution=512, datadir="./data/laion_aesthetics",
    ):
        from dagshub.streaming import DagsHubFilesystem

        self.resolution = resolution
        self.datadir = datadir
        self.fs = DagsHubFilesystem(".", repo_url=self.REPO_URL, branch="main")
        os.makedirs(datadir, exist_ok=True)

        print("Reading labels.tsv from DagsHub...")
        self.samples = []
        with self.fs.open("data/labels.tsv") as tsv:
            for line in tqdm(tsv.readlines()):
                row = line.strip()
                if not row:
                    continue
                img_file, caption, score, url = row.split("\t")
                self.samples.append((img_file, caption))
                if len(self.samples) >= num_samples:
                    break
        print(f"Collected {len(self.samples)} samples.")

    def transform(self, image):
        image = TF.resize(
            image,
            self.resolution,
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        image = TF.center_crop(image, output_size=[self.resolution, self.resolution])
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])
        return image

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_file, text = self.samples[idx]
        local_path = os.path.join(self.datadir, img_file)
        try:
            if not os.path.exists(local_path):
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with self.fs.open(os.path.join("data", img_file), "rb") as src:
                    with open(local_path, "wb") as dst:
                        dst.write(src.read())
            image = Image.open(local_path).convert("RGB")
            image = self.transform(image)
            return image, text
        except Exception:
            return None, None


def collate_skip_none(batch):
    batch = [(img, txt) for img, txt in batch if img is not None]
    if not batch:
        return None, None
    images = torch.stack([b[0] for b in batch])
    texts = [b[1] for b in batch]
    return images, texts


class LatentText2ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
    ):
        self.path = path
        self.np_paths = [
            os.path.join(path, file)
            for file in os.listdir(path)
            if "0" in file and file.endswith(".npy")
        ]
        self.uncond_prompt = np.load(
            os.path.join(path, "uncond_prompt.npy"), allow_pickle=True,
        ).item()["text"]

    def __len__(self):
        return len(self.np_paths)

    def __getitem__(self, idx):
        npy_file = np.load(self.np_paths[idx], allow_pickle=True)
        return torch.from_numpy(npy_file.item()["image"]), torch.from_numpy(
            npy_file.item()["text"],
        )


def encode_prompt(
    prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True,
):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
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
        prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device))[0]

    return prompt_embeds


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compute dataset stat")
    parser.add_argument(
        "--datadir",
        default="./laion/preprocessed_11k/train/",
    )
    parser.add_argument(
        "--save_path",
        default="./dataset/latent_laion_11k/",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="size of image",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2_000_000,
        help="number of samples to use (only for laion/aesthetics_v2_6_5plus)",
    )
    parser.add_argument(
        "--cache_dir",
        default="./data/laion_aesthetics_v2_6_5plus",
        help="local directory to cache downloaded images (only for laion/aesthetics_v2_6_5plus)",
    )

    args = parser.parse_args()

    device = "cuda:0"
    # fp32 for stability
    weight_dtype = torch.float32
    model_id = "XCLIU/2_rectified_flow_from_sd_1_5"

    os.makedirs(args.save_path, exist_ok=True)

    # 2. Load tokenizers from SD-XL checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, subfolder="tokenizer", use_fast=False,
    )

    # 3. Load text encoders
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")

    # 4. Load VAE from SD-XL checkpoint (or more stable VAE)
    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
    )

    if args.datadir == "laion/aesthetics_v2_6_5plus":
        dataset = LaionDagsHubDataset(
            num_samples=args.num_samples, resolution=512, datadir=args.cache_dir,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            collate_fn=collate_skip_none,
        )
    else:
        dataset = Text2ImageDataset(path=args.datadir, resolution=512)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
        )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)

    def compute_embeddings(
        prompt_batch, proportion_empty_prompts, text_encoder, tokenizer, is_train=False,
    ):
        prompt_embeds = encode_prompt(
            prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train,
        )
        return prompt_embeds

    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        proportion_empty_prompts=0,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    uncond_input_ids = tokenizer(
        [""], return_tensors="pt", padding="max_length", max_length=77,
    ).input_ids.to(device)
    uncond_prompt_embeds = text_encoder(uncond_input_ids)[0].detach().cpu().numpy()
    np.save(f"{args.save_path}/uncond_prompt.npy", {"text": uncond_prompt_embeds})

    for i, (image, text) in enumerate(tqdm(dataloader)):
        if image is None:
            continue
        image = image.to(device, non_blocking=True)
        encoded_text = compute_embeddings_fn(text)
        pixel_values = image.to(dtype=weight_dtype)
        if vae.dtype != weight_dtype:
            vae.to(dtype=weight_dtype)

        # encode pixel values with batch size of at most 32
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        latents = latents.to(weight_dtype)
        latents = latents.detach().cpu().numpy()  # (bs, 4, 64, 64)
        encoded_text = encoded_text.detach().cpu().numpy()

        for j in range(len(latents)):
            np.save(
                f"{args.save_path}/{str(i * args.batch_size + j).zfill(12)}.npy",
                {"image": latents[j], "text": encoded_text[j]},
            )
        print(f"Generate batch {i}")

    # test
    debug_idex = list(torch.randint(0, len(dataset), (10,)).numpy())
    data = [
        np.load(f"{args.save_path}/{str(i).zfill(12)}.npy", allow_pickle=True)
        for i in debug_idex
    ]
    sample = torch.stack([torch.from_numpy(x.item()["image"]) for x in data])

    with torch.no_grad():
        rec_image = vae.decode(sample.cuda() / vae.config.scaling_factor).sample
    rec_image = torch.clamp((rec_image + 1.0) / 2.0, 0, 1)
    torchvision.utils.save_image(rec_image, "./rec_debug.jpg")

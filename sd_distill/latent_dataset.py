import os
from glob import glob

import numpy as np
import torch


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

    def get_prompt_null_embed(self):
        return torch.from_numpy(self.uncond_prompt)

    def __getitem__(self, idx):
        npy_file = np.load(self.np_paths[idx], allow_pickle=True)
        return torch.from_numpy(npy_file.item()["image"]), torch.from_numpy(
            npy_file.item()["text"],
        )


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.train = train
        self.transform = transform
        if self.train:
            latent_paths = glob(f"{root}/*.npy")
        else:
            latent_paths = glob(f"{root}/val/*.npy")
        self.data = latent_paths

    def __getitem__(self, index):
        sample = np.load(self.data[index], allow_pickle=True).item()
        target = torch.from_numpy(sample["label"])
        x = torch.from_numpy(sample["input"])
        if self.transform is not None:
            x = self.transform(x)

        return x, target

    def __len__(self):
        return len(self.data)

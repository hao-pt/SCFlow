from glob import glob

import numpy as np
import torch
from torch.utils import data


class LatentDataset(data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.train = train
        self.transform = transform
        if self.train:
            latent_paths = glob(f"{root}/train/*.npy")
        else:
            latent_paths = glob(f"{root}/val/*.npy")
        self.data = latent_paths

    def __getitem__(self, index):
        sample = np.load(self.data[index]).item()
        target = torch.from_numpy(sample["label"])
        x = torch.from_numpy(sample["input"])
        if self.transform is not None:
            x = self.transform(x)

        return x, target

    def __len__(self):
        return len(self.data)

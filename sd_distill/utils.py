import glob
import random
from pathlib import Path

import numpy as np
import torch
from natsort import natsorted
from torch.utils.data import Dataset


# Define a simple prompt-only dataloader
class PromptDataset(Dataset):
    def __init__(self, train_data_dir, datatype, return_dict=True, nsamples=None):
        p = Path(train_data_dir)
        self.datatype = datatype
        self.return_dict = return_dict
        self.nsamples = nsamples
        if self.datatype == "numpy":
            suffix = ".npy"
        elif self.datatype == "torch":
            suffix = ".pt"
        else:
            raise Exception(f"Invalid datatype: {self.datatype}")
        if p.is_dir():
            self.train_data_paths = list(
                natsorted(glob.glob(train_data_dir + f"/*{suffix}")),
            )
            if self.nsamples is not None:
                self.train_data_paths = self.train_data_paths[: self.nsamples]
            self.memmap = None
        else:
            self.train_data_paths = None
            self.memmap = np.load(train_data_dir, mmap_mode="r")
            self.memmap = self.memmap.reshape(-1, 77, 1024)
            if self.nsamples is not None:
                self.memmap = self.memmap[: self.nsamples]

    def _load(self, idx):
        if self.datatype == "numpy":
            if self.train_data_paths is not None:
                return {
                    "prompt_embeds": torch.from_numpy(
                        np.load(self.train_data_paths[idx], allow_pickle=True),
                    ),
                    "prompt_paths": self.train_data_paths[idx],
                }
            return {"prompt_embeds": torch.tensor(self.memmap[idx])}
        if self.datatype == "torch":
            data = torch.load(self.train_data_paths[idx])
            data = {
                "prompt_embeds": data["prompt_embeds"],
            }
            return data
        raise Exception(f"Invalid datatype: {self.datatype}")

    def __len__(self):
        return (
            len(self.train_data_paths)
            if self.train_data_paths is not None
            else len(self.memmap)
        )

    def __getitem__(self, index):
        data = self._load(index)

        if self.return_dict:
            return data
        return data["prompt_embeds"]

    def shuffle(self, *args, **kwargs):
        random.shuffle(self.train_data_paths)
        return self

    def select(self, selected_range):
        self.train_data_paths = [self.train_data_paths[idx] for idx in selected_range]
        return self

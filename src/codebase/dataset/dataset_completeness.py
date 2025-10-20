import os.path

import pandas as pd
import torch
from torch.utils.data import Dataset


class Dataset_completeness(Dataset):
    def __init__(
            self, dataset_path, transform=None, mode=None
    ):
        self.transform = transform
        self.concept_mask = torch.load(os.path.join(dataset_path, f"{mode}_mask_alpha.pt"))
        self.raw_data = torch.load(os.path.join(dataset_path, f"{mode}_tensor_images.pt"))
        self.y = torch.load(os.path.join(dataset_path, f"{mode}_tensor_y.pt"))
        print(f"{mode}_size: {self.raw_data.size()}")
        print(f"{mode}_size: {self.concept_mask.size()}")
        print(f"{mode}_size: {self.y.size()}")

    def __getitem__(self, item):
        image = self.raw_data[item]
        if self.transform:
            image = self.transform(image)
        return image, self.y[item], self.concept_mask[item]

    def __len__(self):
        return self.y.size(0)


class Dataset_completeness_features(Dataset):
    def __init__(self, dataset_path, transform=None, mode=None):
        self.transform = transform
        self.concept_mask = torch.load(os.path.join(dataset_path, f"{mode}_mask_alpha.pt"))
        self.features = torch.load(os.path.join(dataset_path, f"{mode}_tensor_features.pt"))
        self.y = torch.load(os.path.join(dataset_path, f"{mode}_tensor_y.pt"))
        print(f"{mode}_size: {self.features.size()}")
        print(f"{mode}_size: {self.concept_mask.size()}")
        print(f"{mode}_size: {self.y.size()}")

    def __getitem__(self, item):
        return self.features[item], self.y[item], self.concept_mask[item]

    def __len__(self):
        return self.y.size(0)

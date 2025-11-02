import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from torch.nn.functional import one_hot


# ============================================================
#  Base Dataset (mirrors Dataset_cub)
# ============================================================

class Dataset_hatexplain(Dataset):
    def __init__(self, dataset, attributes=None, transform=None, show_text=False):
        self.dataset = dataset  # list of (text, label)
        self.attributes = attributes
        self.transform = transform
        self.show_text = show_text

    def __getitem__(self, idx):
        text, label = self.dataset[idx]
        if self.transform:
            text = self.transform(text)

        if self.attributes is not None:
            attr = self.attributes[idx]
        else:
            attr = torch.zeros(1)  # placeholder if no attributes

        return text, label, attr

    def __len__(self):
        return len(self.dataset)


# ============================================================
#  Dataset with preprocessed tensors (mirrors Dataset_cub_waterbird_landbird)
# ============================================================

class Dataset_hatexplain_split(Dataset):
    def __init__(self, dataset_path, file_name_text, file_name_y, attribute_file_name=None):
        self.text = torch.load(os.path.join(dataset_path, file_name_text))
        self.y = torch.load(os.path.join(dataset_path, file_name_y))
        self.y_one_hot = one_hot(self.y.to(torch.long)).to(torch.float)

        if attribute_file_name:
            self.attributes = torch.load(os.path.join(dataset_path, attribute_file_name))
        else:
            self.attributes = None

        print(self.text.size())
        if self.attributes is not None:
            print(self.attributes.size())
        print(self.y.size())

    def __getitem__(self, idx):
        if self.attributes is not None:
            return self.text[idx], self.attributes[idx], self.y[idx], self.y_one_hot[idx]
        else:
            return self.text[idx], self.y[idx], self.y_one_hot[idx]

    def __len__(self):
        return self.y.size(0)


# ============================================================
#  Dataset for Explainers (mirrors Dataset_cub_for_explainer)
# ============================================================

class Dataset_hatexplain_for_explainer(Dataset):
    def __init__(self, dataset_path, file_name_text, file_name_y, attribute_file_name, raw_data, transform=None):
        self.raw_data = raw_data
        self.transform = transform
        self.text = torch.load(os.path.join(dataset_path, file_name_text))
        self.y = torch.load(os.path.join(dataset_path, file_name_y))
        self.y_one_hot = one_hot(self.y.to(torch.long)).to(torch.float)

        if attribute_file_name:
            self.attributes = torch.load(os.path.join(dataset_path, attribute_file_name))
        else:
            self.attributes = None

        print(self.text.size())
        if self.attributes is not None:
            print(self.attributes.size())
        print(self.y.size())

    def __getitem__(self, idx):
        text = self.raw_data[idx][0]
        if self.transform:
            text = self.transform(text)

        if self.attributes is not None:
            return text, self.text[idx], self.attributes[idx], self.y[idx], self.y_one_hot[idx]
        else:
            return text, self.text[idx], self.y[idx], self.y_one_hot[idx]

    def __len__(self):
        return self.y.size(0)


# ============================================================
#  Main Dataset Class (mirrors Waterbird_LandBird_Final_Dataset)
# ============================================================

class HateXplainDataset(Dataset):
    """
    HateXplain dataset organized in folders:
        data/hatexplain/
            ├── normal/
            ├── offensive/
            └── hatespeech/
        with splits defined in post_id_divisions.json.
    """

    def __init__(self, root_dir, train_transform=None, eval_transform=None, confounder_names=None):
        self.root_dir = root_dir
        print(f"Loading HateXplain from {self.root_dir}")

        # Label mapping (consistent with label_name field)
        self.label_map = {
            'normal': 0,
            'offensive': 1,
            'hatespeech': 2
        }

        # Load split info
        split_path = os.path.join(root_dir, "post_id_divisions.json")
        with open(split_path, 'r', encoding='utf-8') as f:
            split_data = json.load(f)

        print(f"Found splits: {list(split_data.keys())}")

        # Initialize containers
        self.texts = []
        self.tokens = []
        self.y_array = []
        self.attr_array = []  # placeholder (no confounders yet)
        self.filename_array = []
        self.split_array = []

        split_dict = {"train": 0, "val": 1, "test": 2}

        # Iterate through splits and load text
        for split_name, split_idx in split_dict.items():
            post_ids = split_data.get(split_name, [])
            for label_name, label_idx in self.label_map.items():
                folder = os.path.join(root_dir, label_name)

                for pid in post_ids:
                    file_path = os.path.join(folder, f"{pid}.txt")
                    if not os.path.exists(file_path):
                        continue

                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            print(f"Warning: could not parse {file_path}")
                            continue

                    text = data.get("text", "")
                    tokens = data.get("tokens", [])
                    label = data.get("label", label_idx)  # fallback to folder label

                    self.texts.append(text)
                    self.tokens.append(tokens)
                    self.y_array.append(label)
                    self.attr_array.append([0])  # placeholder confounder
                    self.filename_array.append(file_path)
                    self.split_array.append(split_idx)

        self.n_classes = len(self.label_map)
        self.n_confounders = len(self.attr_array[0])
        self.n_groups = self.n_classes * (2 ** self.n_confounders)
        self.target_name = 'label'
        self.confounder_names = confounder_names or ['placeholder_attr']

        self.split_dict = split_dict
        self.train_transform = train_transform
        self.eval_transform = eval_transform

        self.y_array = np.array(self.y_array)
        self.attr_array = np.array(self.attr_array)
        self.split_array = np.array(self.split_array)

        print(f"Loaded {len(self.texts)} posts from HateXplain.")
        print(f"Label distribution: {np.bincount(self.y_array)}")
        print(f"Attr shape: {self.attr_array.shape}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokens[idx]
        y = self.y_array[idx]
        attr = self.attr_array[idx]

        # Apply split-specific transforms
        if self.split_array[idx] == self.split_dict['train'] and self.train_transform:
            text = self.train_transform(text)
        elif self.split_array[idx] in [self.split_dict['val'], self.split_dict['test']] and self.eval_transform:
            text = self.eval_transform(text)

        return text, y, attr, tokens  # returning tokens can help model explainers

    def get_splits(self, splits=('train', 'val', 'test')):
        subsets = {}
        for split in splits:
            assert split in ('train', 'val', 'test')
            mask = self.split_array == self.split_dict[split]
            indices = np.where(mask)[0]
            subsets[split] = Subset(self, indices)
        return subsets


# ============================================================
#  DRO Dataset Wrapper (mirrors DRODatasetFinal)
# ============================================================

class DRODatasetFinal_HateXplain(Dataset):
    def __init__(self, dataset, process_item_fn, n_classes):
        self.dataset = dataset
        self.process_item = process_item_fn
        self.n_classes = n_classes
        y_array = []

        for _, y, *_ in self:
            y_array.append(y)

        self._y_array = torch.LongTensor(y_array)
        self._y_counts = (torch.arange(self.n_classes).unsqueeze(1) == self._y_array).sum(1).float()

    def __getitem__(self, idx):
        if self.process_item is None:
            return self.dataset[idx]
        else:
            return self.process_item(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)

    def class_counts(self):
        return self._y_counts

    def input_size(self):
        for x, y in self:
            return len(x)  # text length proxy

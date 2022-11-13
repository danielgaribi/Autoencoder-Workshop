import math
import random

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate


def load_data(data_dir, batch_size):
    all_files = _list_image_files_recursively(data_dir)

    dataset_train = ImageDataset(all_files, 1000, 40000)
    dataset_val = ImageDataset(all_files, 0, 1000)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

    return loader_train, loader_val


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, image_paths, first_index, last_index):
        super().__init__()
        self.local_images = image_paths[first_index:last_index]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        arr = np.array(pil_image)
        arr = arr.astype(np.float32) / 127.5 - 1
        return np.transpose(arr, [2, 0, 1])

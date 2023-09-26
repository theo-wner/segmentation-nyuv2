import os
import cv2
import numpy as np
import random
import math
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.nn import functional as F
import pytorch_lightning as pl
import config
from utils import visualize_img_mask

class NYUv2DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    # Loads the dataset (not needed data already downloaded)
    def prepare_data(self):
        pass

    def setup(self, stage):
        self.train_dataset = NYUv2Dataset('train')
        self.val_dataset = NYUv2Dataset('val')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=False)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False)


class NYUv2Dataset(Dataset):
    """
    Represents the NYUv2 Dataset
    Example for obtaining an image: image, mask = dataset[0]
    """
    # split is 'train' or 'val', both splits are retrieved from the train set with different transforms
    def __init__(self, split='train'):
        self.root_dir = './data'
        self.image_set = split
        self.ignore_index = config.IGNORE_INDEX

        # Define the path of images and labels
        self.images_dir = os.path.join(self.root_dir, 'image', 'train')
        self.masks_dir = os.path.join(self.root_dir, 'seg13', 'train')

        # Read the list of image filenames
        self.filenames = os.listdir(self.images_dir)

    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, index):
        # Load image and mask
        image = self._load_image(index)
        mask = self._load_mask(index)

        # In case of training, apply data augmentation
        if self.image_set == 'train':
            # Data augmentation
            # Randomly resize image and mask --> Image size changes!
            random_scaler = RandResize(scale=(0.5, 2.0))
            image, mask = random_scaler(image.unsqueeze(0).float(), mask.unsqueeze(0).float())

            # Random Horizontal Flip
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Preprocessing for Random Crop
            if image.shape[1] < 256 or image.shape[2] < 256:
                height, width = image.shape[1], image.shape[2]
                pad_height = max(256 - height, 0)
                pad_width = max(256 - width, 0)
                pad_height_half = pad_height // 2
                pad_width_half = pad_width // 2
                border = (pad_width_half, pad_width - pad_width_half, pad_height_half, pad_height - pad_height_half)
                image = F.pad(image, border, 'constant', 0)
                label = F.pad(label, border, 'constant', 255)

            # Random Crop
            i, j, h, w = transforms.RandomCrop(size=(256, 256)).get_params(image, output_size=(256, 256))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

        # In case of validation, only resize to 256x256
        elif self.image_set == 'val':
            resizer = transforms.Resize(size=(256, 256), interpolation=transforms.InterpolationMode.NEAREST)
            image = resizer(image)
            mask = resizer(mask)

        return image, mask

    def _load_image(self, index):
        image_filename = os.path.join(self.images_dir, self.filenames[index])
        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = TF.to_tensor(image)
        return image
        
    def _load_mask(self, index):
        mask_filename = os.path.join(self.masks_dir, self.filenames[index])
        mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
        mask = TF.to_tensor(mask)
        return mask
    

class RandResize(object):
    """
    Randomly resize image & label with scale factor in [scale_min, scale_max]
    The size of the image gets changed!
    Source: https://github.com/Haochen-Wang409/U2PL/blob/main/u2pl/dataset/augmentation.py
    """
    def __init__(self, scale, aspect_ratio=None):
        self.scale = scale
        self.aspect_ratio = aspect_ratio

    def __call__(self, image, label):
        if random.random() < 0.5:
            temp_scale = self.scale[0] + (1.0 - self.scale[0]) * random.random()
        else:
            temp_scale = 1.0 + (self.scale[1] - 1.0) * random.random()

        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = (self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random())
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)

        scale_factor_w = temp_scale * temp_aspect_ratio
        scale_factor_h = temp_scale / temp_aspect_ratio
        h, w = image.size()[-2:]
        new_w = int(w * scale_factor_w)
        new_h = int(h * scale_factor_h)
        image = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
        label = F.interpolate(label, size=(new_h, new_w), mode="nearest")
        return image.squeeze(), label.squeeze(0)




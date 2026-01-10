import os, os.path as osp
import glob
import cv2
import random
import logging
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets

class MNISTOnlineDataset(Dataset):
    def __init__(self, object_folder, num_max_objects=2000, target_size=32, transform=None):
        self.object_image_paths = glob.glob(osp.join(object_folder, "*.png"))
        random.shuffle(self.object_image_paths)
        if num_max_objects > 0:
            self.object_image_paths = self.object_image_paths[:num_max_objects]
        self.target_size = target_size
        self.transform = transform

    def __getitem__(self, index):
        object_image_path = self.object_image_paths[index]
        object_image = Image.open(object_image_path).convert('L')
        # object_image = np.asarray(Image.open(object_image_path).convert('L')) / 255.
        # object_image = cv2.resize(object_image, (self.target_size, self.target_size))
        if self.transform is not None:
            object_image = self.transform(object_image)
        return object_image

    def __len__(self):
        return len(self.object_image_paths)

class MNISTDataset(Dataset):
    def __init__(self, object_folder, num_max_objects=2000, target_size=32):
        # target size, can be 32 or 240 (for example)
        if osp.isfile(object_folder) and object_folder.endswith(".pkl"):
            self.object_images = pkl.load(open(object_folder, "rb"))
            logging.info('load {} images from {}'.format(len(self.object_images), object_folder))
            self.object_indices = list(range(len(self.object_images)))
            # shuffle and take
            random.shuffle(self.object_indices)
            if num_max_objects > 0:
                self.object_images = self.object_images[self.object_indices[:num_max_objects]]

            # convert to pytorch
            self.object_images = torch.from_numpy(self.object_images).float() / torch.tensor(255.)
            self.object_names = [str(index) + '.png' for index in self.object_indices[:num_max_objects]]
        else:
            self.object_image_paths = glob.glob(osp.join(object_folder, "*.png"))
            random.shuffle(self.object_image_paths)
            if num_max_objects > 0:
                self.object_image_paths = self.object_image_paths[:num_max_objects]
            self.object_images = [np.asarray(Image.open(path).convert('L')) / 255. for path in self.object_image_paths]
            # for i in range(len(self.object_images)):
                # if self.object_images[i].shape[0] != target_size:
                    # self.object_images[i] = cv2.resize(self.object_images[i], (target_size, target_size))
            self.object_images = [torch.from_numpy(object_image).float() for object_image in self.object_images]
            self.object_names = [osp.basename(path) for path in self.object_image_paths]

    def __getitem__(self, index):
        object_image = self.object_images[index]
        return object_image

    def __len__(self):
        return len(self.object_images)


class SimulationDataset(Dataset):
    def __init__(self, object_folder, diffuser_folder, is_train=True,
                 num_max_objects=2000, diffusers_per_epoch=20):
        self.is_train = is_train
        if osp.isfile(object_folder) and object_folder.endswith(".pkl"):
            self.object_images = pkl.load(open(object_folder, "rb"))
            logging.info('load {} images from {}'.format(len(self.object_images), object_folder))
            self.object_indices = list(range(len(self.object_images)))
            # shuffle and take
            random.shuffle(self.object_indices)
            if num_max_objects > 0:
                self.object_images = self.object_images[self.object_indices[:num_max_objects]]
            # convert to pytorch
            self.object_images = torch.from_numpy(self.object_images).float() / 255.
            self.object_names = [str(index) + '.png' for index in self.object_indices[:num_max_objects]]
        else:
            self.object_image_paths = glob.glob(osp.join(object_folder, "*.png"))
            random.shuffle(self.object_image_paths)
            if num_max_objects > 0:
                self.object_image_paths = self.object_image_paths[:num_max_objects]
            self.object_images = [np.asarray(Image.open(path).convert('L')) / 255. for path in self.object_image_paths]
            self.object_images = [torch.from_numpy(object_image).float() for object_image in self.object_images]
            self.object_names = [osp.basename(path) for path in self.object_image_paths]

        self.diffuser_image_paths = glob.glob(osp.join(diffuser_folder, "*.png"))
        self.diffuser_images = [np.asarray(Image.open(path).convert('L')) / 255. for path in self.diffuser_image_paths]
        self.diffuser_images = [np.exp(image * 2j * np.pi) for image in self.diffuser_images]
        logging.info("images read finished, found {} object images and {} diffuser images".format(len(self.object_images), len(self.diffuser_images)))
        self.diffusers_per_epoch = diffusers_per_epoch
        self.diffuser_indices = list(range(0, min(len(self.diffuser_images), diffusers_per_epoch if diffusers_per_epoch > 0 else len(self.diffuser_images))))
        logging.info('diffusers switched to => {}'.format(self.diffuser_indices))
        self.packed_diffusers = torch.stack([torch.from_numpy(self.diffuser_images[index]) for index in self.diffuser_indices], dim=0)

    def get_packed_diffusers(self):
        return self.packed_diffusers

    def switch_diffusers(self):
        num_total_diffusers = len(self.diffuser_images)
        if self.diffuser_indices[-1] >= num_total_diffusers - 1:
            self.diffuser_indices = list(range(0, min(len(self.diffuser_images), self.diffusers_per_epoch if self.diffusers_per_epoch > 0 else len(self.diffuser_images))))
        self.diffuser_indices = list(range(self.diffuser_indices[-1] + 1, min(num_total_diffusers, self.diffuser_indices[-1] + self.diffusers_per_epoch + 1)))
        logging.info('diffusers switched to => {}'.format(self.diffuser_indices))
        self.packed_diffusers = torch.stack(
            [torch.from_numpy(self.diffuser_images[index]) for index in self.diffuser_indices], dim=0)

    def __getitem__(self, index):
        object_image = self.object_images[index]
        return object_image

    def __len__(self):
        return len(self.object_images)
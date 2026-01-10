import os, os.path as osp
import glob
import cv2
import numpy as np
import torch
from PIL import Image



# Randomly sample scattering media.
class DiffuserProvider:
    def __init__(self, diffuser_folder, diffusers_per_epoch=20, rng=None):
        self.diffuser_image_paths = glob.glob(osp.join(diffuser_folder, "*.png"))
        self.diffuser_image_paths = sorted(self.diffuser_image_paths, key=lambda x: int(osp.basename(x).split('.')[0][8:]))
        self.diffuser_images = [np.asarray(Image.open(path).convert('L')) / np.float32(255.) for path in self.diffuser_image_paths]
        self.diffuser_images = [np.exp(image * 2j * np.pi) for image in self.diffuser_images]
        self.diffusers_per_epoch = diffusers_per_epoch
        
        # It sequentially takes scattering media starting from the first one; even if you set a random seed, the order won't change.
        
        # self.diffuser_indices = list(range(0, min(len(self.diffuser_images), diffusers_per_epoch if diffusers_per_epoch > 0 else len(self.diffuser_images))))
        # self.packed_diffusers = torch.stack([torch.from_numpy(self.diffuser_images[index]) for index in self.diffuser_indices], dim=0)
        # print("find {} diffusers in total and provide {} diffusers/epoch".format(len(self.diffuser_images), self.diffusers_per_epoch))
        
        # Randomly select a scattering medium index.
        self.rng = rng  # If not provided, use the global random (controlled by random.seed).
        self.diffuser_indices = self.rng.sample(range(len(self.diffuser_images)), self.diffusers_per_epoch)
        self.packed_diffusers = torch.stack([torch.from_numpy(self.diffuser_images[index]) for index in self.diffuser_indices], dim=0)
        print("find {} diffusers in total and provide {} diffusers/epoch".format(len(self.diffuser_images), self.diffusers_per_epoch))

    def get_packed_diffusers(self):
        return self.packed_diffusers

    def switch_diffusers(self):
        # num_total_diffusers = len(self.diffuser_images)
        # if self.diffuser_indices[-1] >= num_total_diffusers - 1:
        #     self.diffuser_indices = list(range(0, min(len(self.diffuser_images), self.diffusers_per_epoch if self.diffusers_per_epoch > 0 else len(self.diffuser_images))))
        # self.diffuser_indices = list(range(self.diffuser_indices[-1] + 1, min(num_total_diffusers, self.diffuser_indices[-1] + self.diffusers_per_epoch + 1)))
        # self.packed_diffusers = torch.stack(
        #     [torch.from_numpy(self.diffuser_images[index]) for index in self.diffuser_indices], dim=0)
        
        # 随机
        self.diffuser_indices = self.rng.sample(range(len(self.diffuser_images)), self.diffusers_per_epoch)
        self.packed_diffusers = torch.stack(
            [torch.from_numpy(self.diffuser_images[index]) for index in self.diffuser_indices], dim=0)

if __name__ == '__main__':
    train_diffuser_provider = DiffuserProvider("./diffuser_data/train_elight_diffuser")
    packed_diffusers = train_diffuser_provider.get_packed_diffusers()
    print(packed_diffusers.shape)

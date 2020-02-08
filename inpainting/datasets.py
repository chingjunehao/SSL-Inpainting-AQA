import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import cv2
from numpy import unravel_index

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, img_size=129, mask_size=64, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = mode
        self.files = sorted(glob.glob("%s/*.jpg" % root))
        self.files = self.files[:-4000] if mode == "train" else self.files[-4000:]

    def apply_random_mask(self, img):
        """Randomly masks image"""
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part

    def apply_FPP(self, img):
        x1 = 27
        y1 = 27

        x2 = 70
        y2 = 27

        x3 = 27
        y3 = 70

        x4 = 70
        y4 = 70

        masked_part1 = img[:, x1:x1+32, y1:y1+32]
        masked_part2 = img[:, x2:x2+32, y2:y2+32]
        masked_part3 = img[:, x3:x3+32, y3:y3+32]
        masked_part4 = img[:, x4:x4+32, y4:y4+32]

        combined_part1 = np.hstack((masked_part1, masked_part2))
        combined_part2 = np.hstack((masked_part3, masked_part4))

        masked_part = np.concatenate((combined_part1, combined_part2), axis=2)

        masked_img = img.clone()
        masked_img[:, x1:x1+32, y1:y1+32] = 1
        masked_img[:, x2:x2+32, y2:y2+32] = 1
        masked_img[:, x3:x3+32, y3:y3+32] = 1
        masked_img[:, x4:x4+32, y4:y4+32] = 1

        return masked_img, masked_part

    def apply_HSA(self, img):
        img_numpy = img[0].numpy()

        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (_, saliencyMap) = saliency.computeSaliency(img_numpy)
        highest_val_coordinate = unravel_index(saliencyMap.argmax(), saliencyMap.shape)

        x1 = highest_val_coordinate[0]
        y1 = highest_val_coordinate[1]

        if (x1 > 32):
            x1 = x1 - 32
        else:
            x1 = 0

        if (y1 > 32):
            y1 = y1 - 32
        else:
            y1 = 0

        y2, x2 = y1 + self.mask_size, x1 + self.mask_size

        if x2 > self.img_size:
            x1 = x1 - (x2 - self.img_size)
            x2 = self.img_size

        if y2 > self.img_size:
            y1 = y1 - (y2 - self.img_size)
            y2 = self.img_size

        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part

    def apply_HSP(self, img):
        masked_img = img.clone()

        i = (self.img_size - self.mask_size) // 2
        masked_part = img[:, i : i + self.mask_size, i : i + self.mask_size]

        img_numpy = img[0].numpy()

        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (_, saliencyMap) = saliency.computeSaliency(img_numpy)

        top_left = saliencyMap[0:86, 0:86]
        bottom_left = saliencyMap[0:86, 43:129]

        top_right = saliencyMap[43:129, 0:86]
        bottom_right = saliencyMap[43:129, 43:129]

        top_left_values = np.sum(top_left)
        bottom_left_values = np.sum(bottom_left)
        top_right_values = np.sum(top_right)
        bottom_right_values = np.sum(bottom_right)
        coordinate = []

        if top_left_values > bottom_left_values and top_left_values > top_right_values and top_left_values > bottom_right_values:
            masked_part = img[:, 11:75, 11:75]
            masked_img = img.clone()
            masked_img[:, 11:75, 11:75] = 1
            coordinate.append([11,75,11,75])
        elif bottom_left_values > top_left_values and bottom_left_values > top_right_values and bottom_left_values > bottom_right_values:
            masked_part = img[:, 11:75, 54:118]
            masked_img = img.clone()
            masked_img[:, 11:75, 54:118] = 1
            coordinate.append([11,75,54,118])
        elif top_right_values > top_left_values and top_right_values > bottom_left_values and top_right_values > bottom_right_values:
            masked_part = img[:, 54:118, 11:75]
            masked_img = img.clone()
            masked_img[:, 54:118, 11:75] = 1
            coordinate.append([54,118,11,75])
        elif bottom_right_values > top_left_values and bottom_right_values > bottom_left_values and bottom_right_values > top_right_values:
            masked_part = img[:, 54:118, 54:118]
            masked_img = img.clone()
            masked_img[:, 54:118, 54:118] = 1
            coordinate.append([54,118,54,118])

        return masked_img, masked_part, coordinate

    def apply_center_mask(self, img):
        """Mask center part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1

        return masked_img, i

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]).convert('RGB')
        img = self.transform(img)
        if self.mode == "train":
            # For training data perform random mask
            masked_img, aux = self.apply_FPP(img)
        else:
            # For test data mask the center of the image
            masked_img, aux = self.apply_FPP(img)

        return img, masked_img, aux

    def __len__(self):
        return len(self.files)
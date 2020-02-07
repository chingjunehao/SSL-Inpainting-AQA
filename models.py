# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models
import numpy, h5py

from inpainting.datasets import *
from inpainting.models import *

class NIMA_vgg16(nn.Module):
    """Neural IMage Assessment model by Google"""
    def __init__(self, num_classes=10):
        super(NIMA_vgg16, self).__init__()
        base_model = models.vgg16(pretrained=True) 
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=25088, out_features=num_classes), 
            nn.Softmax())

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class inpainting_D_AVA(nn.Module):
    def __init__(self, num_classes=10):
        super(inpainting_D_AVA, self).__init__()
        discriminator = Discriminator(channels=3)

        discriminator.load_state_dict(torch.load("inpainting-pretrained-weights/inpainting-FPP-discriminator.pkl"))

        self.features = discriminator
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=784, out_features=num_classes), 
            nn.Softmax())

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

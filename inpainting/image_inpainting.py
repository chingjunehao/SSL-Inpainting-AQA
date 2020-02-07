import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from datasets import *
from models import *
import torchvision.models as models

import torch.nn as nn
import torch.nn.functional as F
import torch

# Save intermediate images and trained models
os.makedirs("images", exist_ok=True)
os.makedirs("save_models", exist_ok=True)

import lera
lera.log_hyperparams({
        'title': 'G and D Loss'})

parser = argparse.ArgumentParser()
parser.add_argument("--mask_size", type=int, default=64, help="size of random mask")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--img_size", type=int, default=129, help="size of each image dimension")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_epochs", type=int, default=600, help="number of epochs of training")
parser.add_argument("--sample_interval", type=int, default=1, help="interval between image sampling")

opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
patch_h, patch_w = int(opt.mask_size / 2 ** 3), int(opt.mask_size / 2 ** 3)
patch = (1, patch_h, patch_w)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Loss function
adversarial_loss = torch.nn.MSELoss()
pixelwise_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
generator = Generator(channels=opt.channels)
discriminator = Discriminator(channels=opt.channels)


if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Dataset loader
transforms_ = [
    transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
dataloader = DataLoader(
    ImageDataset("/home/viprlab/Documents/aesthetics/project/AROD/arod_train/", transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=12,
)

test_dataloader = DataLoader(
    ImageDataset("/home/viprlab/Documents/aesthetics/project/AROD/arod_val/" , transforms_=transforms_, mode="val"),
    batch_size=12,
    shuffle=True,
    num_workers=1,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def save_sample(batches_done):
    samples, masked_samples, i = next(iter(test_dataloader))
    samples = Variable(samples.type(Tensor))
    masked_samples = Variable(masked_samples.type(Tensor))
    
    x1 = 27
    y1 = 27

    x2 = 70
    y2 = 27

    x3 = 27
    y3 = 70

    x4 = 70
    y4 = 70

    # Generate inpainted image
    gen_mask = generator(masked_samples)

    filled_samples = masked_samples.clone()
    filled_samples[:, :, x1:x1+32, y1:y1+32] = gen_mask[:, :, 0:32, 0:32]
    filled_samples[:, :, x2:x2+32, y2:y2+32] = gen_mask[:, :, 32:64, 0:32]
    filled_samples[:, :, x3:x3+32, y3:y3+32] = gen_mask[:, :, 0:32, 32:64]
    filled_samples[:, :, x4:x4+32, y4:y4+32] = gen_mask[:, :, 32:64, 32:64]


    # Save sample
    sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
    save_image(sample, "images/%d.png" % batches_done, nrow=6, normalize=True)



# ----------
#  Training
# ----------
total_g_loss = 0
total_d_loss = 0
prev_g = 10000
prev_d = 10000
for epoch in range(opt.n_epochs):
    for i, (imgs, masked_imgs, masked_parts) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)

        # Configure input
        imgs = Variable(imgs.type(Tensor))
        masked_imgs = Variable(masked_imgs.type(Tensor))
        masked_parts = Variable(masked_parts.type(Tensor))
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()
        # Generate a batch of images
        gen_parts = generator(masked_imgs)
        # Adversarial and pixelwise loss
        g_adv = adversarial_loss(discriminator(gen_parts), valid)
        g_pixel = pixelwise_loss(gen_parts, masked_parts)
        # Total loss
        g_loss = 0.001 * g_adv + 0.999 * g_pixel

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(masked_parts), valid)
        fake_loss = adversarial_loss(discriminator(gen_parts.detach()), fake)


        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f, pixel: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_adv.item(), g_pixel.item())
        )
        lera.log
        # Generate sample at sample interval
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_sample(batches_done)

        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()
        lera.log("g_loss", g_loss.item())
        lera.log("d_loss", d_loss.item())

    avg_d_loss = total_d_loss / len(dataloader)
    avg_g_loss = total_g_loss / len(dataloader)
    lera.log("avg_g_loss", avg_g_loss)
    lera.log("avg_d_loss", avg_d_loss)
    print("Average g_loss: %f, Average d_loss: %f" % (avg_g_loss, avg_d_loss))

    if avg_d_loss < prev_d:
        prev_d = avg_d_loss
        torch.save(discriminator.state_dict(), os.path.join("/media/viprlab/01D31FFEF66D5170/Junehao/save_models", 'discriminator_epoch-%d.pkl' % (epoch + 1)))

    if avg_g_loss < prev_g:
        prev_g = avg_g_loss
        torch.save(generator.state_dict(), os.path.join("/media/viprlab/01D31FFEF66D5170/Junehao/save_models", 'generator_epoch-%d.pkl' % (epoch + 1)))

    total_g_loss = 0
    total_d_loss = 0



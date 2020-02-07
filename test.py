import argparse
import os
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from models import *

from scipy.stats import spearmanr, pearsonr

def single_emd_loss(p, q, r=1):
    assert p.shape == q.shape, "Length of the two distribution must be the same"
    length = p.shape[0]
    emd_loss = 0.0
    for i in range(1, length + 1):
        emd_loss += torch.abs(sum(p[:i] - q[:i])) ** r
    return (emd_loss / length) ** (1. / r)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='path to pretrained model', default="path/to/pretrained/model")
parser.add_argument('--test_csv', type=str, help='test csv file', default='/path/to/csv/file')
parser.add_argument('--test_images', type=str, help='path to folder containing images', default='/path/to/images/file')
args = parser.parse_args()

model = inpainting_D_AVA()

try:
    model.load_state_dict(torch.load(args.model))
    print('successfully loaded model')
except:
    raise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

model.eval()

test_transform = transforms.Compose([
    transforms.Scale(256), 
    transforms.RandomCrop(224), 
    transforms.ToTensor()
    ])


test_imgs = [f for f in os.listdir(args.test_images)]

test_df = pd.read_csv(args.test_csv, header=None)

predicted_mean = []
groundTruth_mean = []
predicted_std = []
groundTruth_std = []
predicted = 0
emd_loss = []

for i, img in enumerate(test_imgs):
    im = Image.open(os.path.join(args.test_images, img))
    imt = test_transform(im)
    imt = imt.unsqueeze(dim=0)
    imt = imt.to(device)
    with torch.no_grad():
        out = model(imt)
    out = out.view(10, 1)
    print(out)
    
    mean, std = 0.0, 0.0
    # Get mean of predicted score
    for j, e in enumerate(out, 1):
        mean += j * e

    # Get standard deviation of predicted score
    for k, e in enumerate(out, 1):
        std += (e * (k - mean) ** 2) ** (0.5)

    gt = test_df[test_df[0] == int(img.split('.')[0])].to_numpy()[:, 1:].reshape(10, 1)
    
    # Get ground truth mean and ground truth standard deviation 
    gt_mean, gt_std = 0.0, 0.0
    for l, e in enumerate(gt, 1):
        gt_mean += l * e

    for k, e in enumerate(gt, 1):
        gt_std += (e * (k - gt_mean) ** 2) ** (0.5)

    gt = torch.from_numpy(gt).to(device).float()
    sel = single_emd_loss(gt, out)
    emd_loss.append(sel)

    predicted_mean.append(mean.cpu().numpy()[0])
    groundTruth_mean.append(gt_mean[0])
    predicted_std.append(std.cpu().numpy()[0])
    groundTruth_std.append(gt_std[0])

    print(img.split('.')[0] + ' mean: %.3f | std: %.3f | GT_mean: %.3f | GT_std: %.3f' % (mean, std, gt_mean, gt_std))

    if (mean >= 5 and gt_mean >= 5) or (mean < 5 and gt_mean < 5):
        predicted += 1


print("Accuracy on binary classes:")
print(predicted/20000)

c = np.corrcoef(predicted_mean, groundTruth_mean)[0, 1]
print(np.corrcoef(predicted_mean, groundTruth_mean))
print("LCC on predicted mean and ground truth mean:")
print(c)

rho, _ = spearmanr(predicted_mean, groundTruth_mean)
print(spearmanr(predicted_mean, groundTruth_mean))
print("SRCC on predicted mean and ground truth mean:")
print(rho)

c = np.corrcoef(predicted_std, groundTruth_std)[0, 1]
print(np.corrcoef(predicted_std, groundTruth_std))
print("LCC on predicted stanndard deviation and ground truth standard deviation:")
print(c)

rho, _ = spearmanr(predicted_std, groundTruth_std)
print(spearmanr(predicted_std, groundTruth_std))
print("SRCC on predicted stanndard deviation and ground truth standard deviation:")
print(rho)

avg_emd_loss = sum(emd_loss)/len(emd_loss)
print("Average EMD loss")
print(avg_emd_loss)


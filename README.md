# Learning image aesthetics by learning inpainting

## Overview of the project
![teaser](/assets/teaser.png)
## Architecture 
![GAN](/assets/GAN_architecture.png)
## Methods of image inpainting
![inpainting methods](/assets/inpainting-methods.png)

Ways to apply the masking are in ```inpainting/datasets.py```

## Getting started for Self-supervised learning with image inpainting
### Step 1) Installation of packages with pip
```
pip install -r requirements
```
### Step 2) Change of paths
May change the path to dataset, pre-trained models as well as the architecture of models to the one you want to.

### Step 3) Start training 
```
python image_inpainting.py
```

## Aesthetics quality assessment task training on AVA
### Step 1) Download AVA
[link](http://academictorrents.com/details/71631f83b11d3d79d8f84efe0a7e12f0ac001460)

### Step 2) Download CSV file (AVA1 and AVA2)
[gdrive link](https://drive.google.com/open?id=1kSjcOHagdcyqmwUbH7bpAWvioOWQjrHU)

### Step 3) Change of paths
May change the path to dataset, pre-trained models as well as the architecture of models to the one you want to.

### Step 4) Start training 
```
python main.py
```

## Acknowledgement
This code borrows heavily from [kentsyx](https://github.com/kentsyx/Neural-IMage-Assessment) and [eriklindernoren](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/context_encoder/context_encoder.py) repositories.




#!/bin/bash
# First set of pix2pix experiments

python train.py --name ff_linear_1 --model feedforward --netG linearfilt --niter 200 --niter_decay 50

python train.py --name ff_resnet9_1 --model feedforward --netG resnet_9blocks --niter 150 --niter_decay 100 
python train.py --name ff_resnet6_1 --model feedforward --netG resnet_6blocks --niter 150 --niter_decay 100 
python train.py --name ff_unet256_1 --model feedforward --netG unet_256 --niter 150 --niter_decay 100

python train.py --name ff_resnet9_2 --model feedforward --netG resnet_9blocks --regression_loss L2 --niter 150 --niter_decay 100 
python train.py --name ff_resnet6_2 --model feedforward --netG resnet_6blocks --regression_loss L2 --niter 150 --niter_decay 100 
python train.py --name ff_unet256_2 --model feedforward --netG unet_256 --regression_loss L2 --niter 150 --niter_decay 100

python train.py --name pix2pix_resnet9_final --model pix2pix --netG resnet_9blocks --niter 150 --niter_decay 100 --lambda_L1 100
python train.py --name pix2pix_resnet6_final --model pix2pix --netG resnet_6blocks --niter 150 --niter_decay 100 --lambda_L1 100
python train.py --name pix2pix_unet256_final --model pix2pix --netG unet_256 --niter 150 --niter_decay 100 --lambda_L1 100

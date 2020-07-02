#!/bin/bash
# Tuning final models for paper - experiment set 1

python train.py --name repair_resnet9_low3 --model repair --netG resnet_9blocks --niter 50 --niter_decay 50 --lambda_L1 1
python train.py --name repair_resnet9_low4 --model repair --netG resnet_9blocks --niter 50 --niter_decay 50 --lambda_L1 10
python train.py --name repair_resnet9_low5 --model repair --netG resnet_9blocks --niter 50 --niter_decay 50 --lambda_L1 100 # Test set model

python train.py --name pix2pix_resnet9_low4 --model pix2pix --netG resnet_9blocks --niter 50 --niter_decay 50 --lambda_L1 1
python train.py --name pix2pix_resnet9_low5 --model pix2pix --netG resnet_9blocks --niter 50 --niter_decay 50 --lambda_L1 10
python train.py --name pix2pix_resnet9_low6 --model pix2pix --netG resnet_9blocks --niter 50 --niter_decay 50 --lambda_L1 50
python train.py --name pix2pix_resnet9_low7 --model pix2pix --netG resnet_9blocks --niter 50 --niter_decay 50 --lambda_L1 100

python train.py --name pix2pix_unet256_low2 --model pix2pix --netG unet_256 --niter 50 --niter_decay 50 --lambda_L1 1
python train.py --name pix2pix_unet256_low3 --model pix2pix --netG unet_256 --niter 50 --niter_decay 50 --lambda_L1 10
python train.py --name pix2pix_unet256_low4 --model pix2pix --netG unet_256 --niter 50 --niter_decay 50 --lambda_L1 50
python train.py --name pix2pix_unet256_low5 --model pix2pix --netG unet_256 --niter 50 --niter_decay 50 --lambda_L1 100
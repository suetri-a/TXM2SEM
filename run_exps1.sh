#!/bin/bash
# First set of pix2pix experiments

python train.py --name pix2pix_resnet9_1 --model pix2pix --netG resnet_9blocks --niter 200 --niter_decay 0 --lambda_L1 100
python train.py --name pix2pix_resnet6_1 --model pix2pix --netG resnet_6blocks --niter 100 --niter_decay 0 --lambda_L1 100
python train.py --name pix2pix_unet256_1 --model pix2pix --netG unet_256 --niter 100 --niter_decay 0 --lambda_L1 100

python train.py --name pix2pix_resnet9_2 --model pix2pix --netG resnet_9blocks --niter 100 --niter_decay 0 --lambda_L1 500
python train.py --name pix2pix_resnet6_2 --model pix2pix --netG resnet_6blocks --niter 100 --niter_decay 0 --lambda_L1 500
python train.py --name pix2pix_unet256_2 --model pix2pix --netG unet_256 --niter 100 --niter_decay 0 --lambda_L1 500

python train.py --name pix2pix_resnet9_3 --model pix2pix --netG resnet_9blocks --niter 100 --niter_decay 0 --lambda_L1 1000
python train.py --name pix2pix_resnet6_3 --model pix2pix --netG resnet_6blocks --niter 100 --niter_decay 0 --lambda_L1 1000
python train.py --name pix2pix_unet256_3 --model pix2pix --netG unet_256 --niter 100 --niter_decay 0 --lambda_L1 1000
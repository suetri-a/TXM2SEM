#!/bin/bash
# Low L1 penalty models

# python train.py --name pix2pix_resnet9_low1 --model pix2pix --netG resnet_9blocks --niter 80 --niter_decay 20 --lambda_L1 1
# python train.py --name pix2pix_resnet9_low2 --model pix2pix --netG resnet_9blocks --niter 80 --niter_decay 20 --lambda_L1 10
# python train.py --name pix2pix_resnet9_low3 --model pix2pix --netG resnet_9blocks --niter 80 --niter_decay 20 --lambda_L1 50

python train.py --name pix2pix_unet_low1 --model pix2pix --netG unet_256 --niter 80 --niter_decay 20 --lambda_L1 10

#!/bin/bash
# Tuning final models for paper - experiment set 2

python train.py --name pix2pix_resnet9_low5 --model pix2pix --netG resnet_9blocks --niter 50 --niter_decay 50 --lambda_L1 10

python train.py --name pix2pix_resnet6_low1 --model pix2pix --netG resnet_6blocks --niter 50 --niter_decay 50 --lambda_L1 1
python train.py --name pix2pix_resnet6_low2 --model pix2pix --netG resnet_6blocks --niter 50 --niter_decay 50 --lambda_L1 10
python train.py --name pix2pix_resnet6_low3 --model pix2pix --netG resnet_6blocks --niter 50 --niter_decay 50 --lambda_L1 50
python train.py --name pix2pix_resnet6_low4 --model pix2pix --netG resnet_6blocks --niter 50 --niter_decay 50 --lambda_L1 100

python train.py --name ff_resnet9_3 --model feedforward --netG resnet_9blocks --niter 50 --niter_decay 50

python train.py --name ff_linear_3 --model feedforward --netG linearfilt --niter 50 --niter_decay 50

python train.py --name repair_resnet9_low5_semonly --model repair --netG resnet_9blocks --niter 50 --niter_decay 50 --lambda_L1 100 --sem_only True

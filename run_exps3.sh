#!/bin/bash
# Regularized models

python train.py --name ff_resnet9_reg1 --model feedforward --netG resnet_9blocks --niter 80 --niter_decay 20 --weight_decay 1e-6
python train.py --name ff_resnet9_reg2 --model feedforward --netG resnet_9blocks --niter 80 --niter_decay 20 --weight_decay 1e-5
python train.py --name ff_resnet9_reg3 --model feedforward --netG resnet_9blocks --niter 80 --niter_decay 20 --weight_decay 1e-4

python train.py --name pix2pix_resnet9_reg1 --model pix2pix --netG resnet_9blocks --niter 80 --niter_decay 20 --lambda_L1 100 --weight_decay 1e-6
python train.py --name pix2pix_resnet9_reg2 --model pix2pix --netG resnet_9blocks --niter 80 --niter_decay 20 --lambda_L1 100 --weight_decay 1e-5
python train.py --name pix2pix_resnet9_reg3 --model pix2pix --netG resnet_9blocks --niter 80 --niter_decay 20 --lambda_L1 100 --weight_decay 1e-4

python train.py --name pix2pix_resnet9_reg4 --model pix2pix --netG resnet_9blocks --niter 80 --niter_decay 20 --lambda_L1 1000 --weight_decay 1e-6
python train.py --name pix2pix_resnet9_reg5 --model pix2pix --netG resnet_9blocks --niter 80 --niter_decay 20 --lambda_L1 1000 --weight_decay 1e-5
python train.py --name pix2pix_resnet9_reg6 --model pix2pix --netG resnet_9blocks --niter 80 --niter_decay 20 --lambda_L1 1000 --weight_decay 1e-4

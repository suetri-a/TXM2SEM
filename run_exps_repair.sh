#!/bin/bash
# Low L1 penalty models

python train.py --name repair_resnet9_low1 --model repair --netG resnet_9blocks --niter 80 --niter_decay 20 --lambda_L1 1
python train.py --name repair_resnet9_lrlow1 --model repair --netG resnet_9blocks --niter 80 --niter_decay 20 --lambda_L1 1 --lr 0.0001
python train.py --name repair_resnet9_low2 --model repair --netG resnet_9blocks --niter 80 --niter_decay 20 --lambda_L1 10
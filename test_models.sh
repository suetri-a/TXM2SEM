#!/bin/bash
# Test set models

python test.py --name repair_resnet9_low5 --model repair --netG resnet_9blocks

python test.py --name pix2pix_resnet9_low4 --model pix2pix --netG resnet_9blocks

python test.py --name pix2pix_resnet6_low1 --model pix2pix --netG resnet_6blocks

python test.py --name pix2pix_unet256_low3 --model pix2pix --netG unet_256

python test.py --name ff_resnet9_3 --model feedforward --netG resnet_9blocks

python test.py --name ff_linear_3 --model feedforward --netG linearfilt 

python test.py --name repair_resnet9_low5_semonly --model repair --netG resnet_9blocks --sem_only True

python train.py --name repair_resnet9_low5 --model repair --netG resnet_9blocks 

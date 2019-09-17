#!/bin/bash
# First set of pix2pix experiments

python test.py --name ff_linear_1 --model feedforward --netG linearfilt

python test.py --name ff_resnet9_1 --model feedforward --netG resnet_9blocks 
python test.py --name ff_resnet6_1 --model feedforward --netG resnet_6blocks
python test.py --name ff_unet256_1 --model feedforward --netG unet_256 

python test.py --name pix2pix_resnet9_final --model pix2pix --netG resnet_9blocks
python test.py --name pix2pix_resnet6_final --model pix2pix --netG resnet_6blocks
python test.py --name pix2pix_unet256_final --model pix2pix --netG unet_256 

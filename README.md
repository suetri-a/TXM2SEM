# TXM2SEM Image Translation

This repository contains a framework for training and evaluating image-to-image translation models for translating transmission X-ray microscopy (TXM) images to focused ion beam milled-scanning electron microscopy (FIB-SEM) images. The code presented is mainly from the [original pix2pix code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) published by [Jun-Yan Zhu](https://github.com/junyanz) and [Taesung Park](https://github.com/taesung), and supported by [Tongzhou Wang](https://ssnl.github.io/). The relevant papers are:

- Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. [Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz/)\*,  [Taesung Park](https://taesung.me/)\*, [Phillip Isola](https://people.eecs.berkeley.edu/~isola/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In ICCV 2017. (* equal contributions) [[Bibtex]](https://junyanz.github.io/CycleGAN/CycleGAN.txt)

- Image-to-Image Translation with Conditional Adversarial Networks. [Phillip Isola](https://people.eecs.berkeley.edu/~isola), [Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz), [Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In CVPR 2017. [[Bibtex]](http://people.csail.mit.edu/junyanz/projects/pix2pix/pix2pix.bib)

We have modified and simplified their original framework including implementation of new models and data loaders, modification of the visualization pipeline, and removal of code for unpaired image translation. Below, we describe the problem of interest and instructions for applying this code. 


## Problem Overview

Micro- and nanoscale characterization is essential for understanding energy technologies including subsurface CO<sub>2</sub> and H<sub>2</sub> storage and fossil energy recovery. A key challenge in reservoir rock characterization is obtaining high-resolution images of the rock fabric while preserving the sample for further experimentation. Image modalities such as FIB-SEM yield high-resolution/high-contrast images that contain information about the structure and mineral composition of the rock sample, but necessarily destroy the sample during image acquisition, thus precluding any further experimentation on the sample. Meanwhile, non-destructive imaging modalities such as TXM are able to preserve samples during image acquisition but acquire much lower resolution/contrast images. An ideal imaging setup would have the resolution and contrast of FIB-SEM while preserving the sample. 

We address this by using 2D paired multimodal images to train 2D-to-2D image models to predict FIB-SEM images from input TXM. We employ image translation and single image super-resolution (SISR) models to predict FIB-SEM images from TXM images. All models train a generator network <img src="https://render.githubusercontent.com/render/math?math=\hat{S} = G_\theta(T)"> to map a TXM image <img src="https://render.githubusercontent.com/render/math?math=T"> to a FIB-SEM image <img src="https://render.githubusercontent.com/render/math?math=S">. In the following section, we outline the specific of these models.


## Models
The TXM and FIB-SEM images have the same machine resolution, but in practice the effective resolution of TXM images is lower than that of the FIB-SEM images. For this reason, we test two main families of models:
1. **Image translation:** treats mapping TXM to SEM images as a style transfer problem between two image domains. These models map a full-resolution TXM image <img src="https://render.githubusercontent.com/render/math?math=T \in \mathbb{R}^{N \times N}"> to a full-resolution FIB-SEM image <img src="https://render.githubusercontent.com/render/math?math=S \in \mathbb{R}^{N\times N}">.
2. **Single image super-resolution:** treats mapping TXM to SEM images as an upsampling problem. These models are designed to map synthetically-downsampled TXM images <img src="https://render.githubusercontent.com/render/math?math=T \in \mathbb{R}^{N/\gamma \times N/\gamma}"> to a full-resolution FIB-SEM image <img src="https://render.githubusercontent.com/render/math?math=S \in \mathbb{R}^{N\times N}">. Here, <img src="https://render.githubusercontent.com/render/math?math=\gamma"> is the super-resolution factor (2 or 4 in the models tested) determining the assumed difference in effective resolution between the TXM and SEM images. We used a model closely based onthat from [Ledig et al. 2016](https://arxiv.org/abs/1609.04802). 

For both models, we test out feedforward convolutional neural network (CNN) and conditional generative adversarial network (CGAN) versions of the models. The training objective for these models are:
* **Feedforward CNN:** <img src="https://render.githubusercontent.com/render/math?math=\underset{\theta}{\text{minimize}} \sum_{i=1}^n ||G_\theta(T_i) - S_i||_1">
* * **CGAN:** <img src="https://render.githubusercontent.com/render/math?math=\underset{\theta}{\text{minimize}} \sum_{i=1}^n \mathcal{L}_{\text{GAN}}(\theta) %2B \lambda ||G_\theta(T_i) - S_i||_1"> where <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{\text{GAN}}"> is either the vanilla or Wasserstein GAN objective and <img src="https://render.githubusercontent.com/render/math?math=\lambda"> is a hyperparameter adjusting the image similarity penalty loss term in the GAN objective. The addition of the GAN loss term encourages stylistic similarity between the generated SEM images. 

We also implement an original z-direction continuity loss term to encourage continutiy between adjacent TXM input images. The multimodal image dataset used to train the models contains only paired 2D images, so the challenge is reconstructing 3D volumes with only training data. One solution is to train a generator network that is continuous in the input TXM image. This can be done using a z-regularization term of the form:

<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{\text{z-regularization}} = \left| \left| \frac{ \partial \hat{S}}{ \partial T} \right| \right|_2^2">

Using this, we can train the models as:

<img src="https://render.githubusercontent.com/render/math?math=\underset{\theta}{\text{minimize}} \mathcal{L}_{\text{Original}} %2B \lambda_{\text{reg}} \mathcal{L}_{\text{z-regularization}}">

We use the efficient Jacobian penalty implementation provided by [Hoffman et al. (2019)](https://arxiv.org/abs/1908.02729). 


## Usage

The model is run through a command line interface. We strongly recommend installing [Anaconda](https://www.anaconda.com/products/individual) as it includes most of the packages needed for this code base, and the ``conda`` package management system is able to install almost everything required. 

### Setup

To begin, first install the dependencies listed here. The code requires the following packages listed below. Unless otherwise stated, these packages can be installed using ``conda`` or ``pip``
- ``torch``: install from the [PyTorch website](https://pytorch.org/)
- ``skimage``
- ``PIL``
- ``re``
- ``glob``
- ``jacobian``: install from [the author's repository](https://github.com/facebookresearch/jacobian_regularizer)
- ``dominate``
- ``visdom``: install with ``pip``

### Data 

Image data is expected to be stored using the following file structure for data loaders:
```
./images/
|
+--train/
|  +--txm/
|     +--[image number with three digits e.g. 000 or 015].tif
|  +--sem/
|     +--[image number].tif
|  +--charge/
|     +--[image number].tif
|  +--lowdensity/
|     +--[image number].tif
|  +--highdensity/
|     +--[image number].tif
|
+--val/
|  + ...
+--test/
|  + ...
|  +--txm_full_stack/
|     +--[image number in z-axis order, unrelated to the numbers of the aligned slices].tif
```

The data loaders rely on folders and filenames appearing in this specific form. If the images are not places in the correct folder, the dataloader will not be able to find them. Aligned image slices should appear with the same file names in the ``txm``, ``sem``, ``charge``, ``lowdensity``, and ``highdensity`` folders. This is how the code is able to track which slices are aligned with which. The ``txm_full_stack`` folder in the ``test/`` contains TXM images from a contiguous volume where each slice is numbered according to its slice number in the z-axis. 

Dataset files can be found in the ``./data/`` folder. The framework implements four data loaders depending on the application. The specific dataset to use is selected with the ``--dataset`` option during training and testing. 

**TXM2SEM**

The ``txm2sem_dataset.py`` file contains the main dataset for this framework. 

Command line options specific to this data loader are:
- ``--aligned``: optionally use aligned or unaligned image patches
- ``--eval_mode``: determines whether dataset has fixed indices (for evaluation) or random (for sampling during training)
- ``--patch_size``: image patch size when performing subsampling
- ``--txm_dir``: TXM image directory. This and all image directories below can be controlled using command line options but this is highly discouraged.
- ``--sem_dir``: SEM image directory
- ``--charge_dir``: charge region segmentation directory
- ``--lowdens_dir``: low density region segmentation directory
- ``--highdens_dir``: high density region segmentation directory
- ``--num_train``: number of datapoints in each training epoch

**Image Repair**

The ``image_repair_data.py`` loader functions very similarly to the ``txm2sem_dataset.py`` loader, with the only major difference being the form of the data output during sampling. Command line options specific for this data loader are the same as for the ``txm2sem_dataset.py`` loader above.

**TXM2SEM3D**

The ``txm2sem3d_dataset.py`` file contains a short dataloader to load TXM image volumes from the test set folder. TXM images in the ``/test/txm_full_stack/`` folder should be full image slices (uncropped). The code uses the ``x_ind`` and ``y_ind`` arguments as the top-left corner of the image patch from each slice. In this way, the subvolume to process can be controlled from the command line. 

Command line options specific to this data loader are:
- ``--patch_size``: image patch size when performing subsampling
- ``--save_name``: directory to store the saved volume in the results folder
- ``--x_ind``: x-index for sampling image patches
- ``--y_ind``: y-index for sampling image patches 


### Training a model

The model is trained using the ``train.py`` script. For example, you can train a model using the following code: 

``python train.py --name srcnn_imgreg --model srcnn --netG sr_resnet_9blocks --niter 50 --niter_decay 25 --downsample_factor 2 --patch_size 128 --lambda_img_grad 1e-4``

This will train a SRCNN model with a 9 block SR-ResNet, decay the learning rate after 25 epochs, assume a 2x downsampling factor in the TXM images, train with 128x128 image patches, and uses a 10<sup>-4</sup> Jacobian penalty parameter. Each model has its own set of commands: 

**Feedforward CNN:** ``feedforward_model.py``
* ``--lambda_regression``: weight for the regression loss
* ``--regression_loss``: loss type for the regression (L2 or L1)
* ``--lambda_img_grad``: weight to image gradient penalty

**SR-CNN:** ``srcnn_model.py``
* ``--lambda_regression``: weight for the regression loss
* ``--regression_loss``: loss type for the regression (L2 or L1)
* ``--lambda_img_grad``: weight to image gradient penalty

**pix2pix CGAN:** ``pix2pix_model.py``
* ``--lambda_L1``: weight for L1 loss
* ``--lambda_img_grad``: weight to image gradient penalty
* ``--lambda_gp``: weight for gradient penalty in wgangp loss. Default to 0, must be entered manually as ``--gan_mode wgangp --lambda_gp 1e1`` (recommended value is 10).

**SRGAN:** ``srgan_model.py``
* ``--lambda_L1``: weight for L1 loss
* ``--lambda_img_grad``: weight to image gradient penalty
* ``--lambda_gp``: weight for gradient penalty in wgangp loss
* ``--d_condition``: if flag is passed, input TXM image will be passed to discriminator network along with generated SEM image. (This differs from the original SR-GAN model which only passes the generated image into the discriminator.) 

*Warning:* running ``train.py`` will clear all files in the ``./results/[model name]/`` directory and erase any previous results. Please be cautious not to delete any results by accidentally running the training script instead of the test script.


### Loading and testing a model 

The framework is automatically able to load a model based on the model name. To load a model, 

```
./checkpoints
|
+--/[model name]
   |
   +--[epoch #]_net_[net name].pth
   +--latest_net_[net name].pth

```

To test and evaluate a TXM-to-SEM image translation model, use:

``python test.py --name srgan_example --model srgan --netG sr_resnet_9blocks --downsample_factor 4 --patch_size 128``

During testing, The code will look in the checkpoints folder for a ``srgan_example`` folder, load the ``latest_net_G.pth`` in to a 4x SR-ResNet 9 block network, and evaluate the image patches and similarity metrics for the dataset. 

To evaluate for 3D volume translation, you must use the ``txm2sem3d`` dataset mode. Besides this, the command line argument is very similar:

``python test.py --name srgan_example --model srgan --netG sr_resnet_9blocks --downsample_factor 4 --patch_size 128 --dataset_mode txm2sem3d --x_ind 280 --y_ind 165 --save_name sem_predicted_volume ``

The framework will output the results in the following file structure:

```
./results 
|
+--/[model name]
   |
   +--/charge (charge region mask)
      +--[image patch number with three digits].png
      +--...
   +--/highdensity (high density region mask)
      +--...
   +--/lowdensity (low region mask)
      +--...
   +--/sem (ground truth SEM image)
      +--...
   +--/sem_fake (predicted SEM image)
      +--...
   +--/txm (input TXM image)
      +--...
   +--/volume_pred_test (predicted FIB-SEM volume slices)
      +--...
   +--/volume_txm_test (input TXM volume image slices
      +--...
   +--eval_metrics.txt (summary of evaluation metrics)
```

The code saves the test set images every time it is run, but it is configured to produce the same test set subsampled image patches given the same test set slices. *Note:* the 3D dataset will also output quantitative image similarity metrics, but these results should be ignored during 3D volume generation. These results only appear because of difficulty changing the implementation.

We have two models from our research available for download: 
* SR GAN 4x with no image gradient penalty \[[Model](https://stanford.box.com/s/urj3uqwymkmx499zipu3w4ef99beypk1)\]
  * ``` python train.py --name srgan_no_reg --model srgan --netG sr_resnet_9blocks --niter 50 --niter_decay 25 --downsample_factor 4 --patch_size 128 --lambda_L1 100 ```
  * ``` python test.py --name srgan_no_reg --model srgan --netG sr_resnet_9blocks --downsample_factor 4 --patch_size 128 ```
  * ```  python test.py --name srgan_no_reg --model srgan --netG sr_resnet_9blocks --downsample_factor 4 --patch_size 128 --dataset_mode txm2sem3d --x_ind [x index] --y_ind [y index] --save_name [save name] ```
* SR GAN 4x with image gradient penalty \[[Model](https://stanford.box.com/s/up51dlasj7auvqc2g98fdk4iwlluqwi5)\]
  * ``` python train.py --name srgan_imgreg --model srgan --netG sr_resnet_9blocks --niter 50 --niter_decay 25 --downsample_factor 4 --patch_size 128 --lambda_L1 1000 --lambda_img_grad 1e-4 ```
  * ``` python test.py --name srgan_imgreg --model srgan --netG sr_resnet_9blocks --downsample_factor 4 --patch_size 128 ```
  * ``` python test.py --name srgan_imgreg --model srgan --netG sr_resnet_9blocks --downsample_factor 4 --patch_size 128 --dataset_mode txm2sem3d --x_ind [x index] --y_ind [y index] --save_name [save name] ```

These models are ready to use or to test that your framework and data loader are working properly. 


### Command Line Interface Summary

Here we summarize the command line arguments used across all models. 

**General commands**

Basic parameters:
* ``--dataroot``: path to images. Should have subfolders ``training``, ``val``, and ``test``, which should each have subfolders ``TXM and ``SEM``
* ``--name``: ame of the experiment. It decides where to store samples and models.
* ``--gpu_ids``: gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU.
* ``--checkpoints_dir``: models are saved here (default: ``./checkpoints``).
* ``--downsample_factor``: factor by which to downsample synthetic data. Used with SISR models.

Model parameters:
* ``--model``: chooses which model to use. \[feedforward | srcnn | pix2pix | srgan \]
* ``--input_nc``: \# of input image channels (default=1, should not need to be changed)
* ``--output_nc``: \# of output image channels (default=1, should not need to be changed)
* ``--ngf``: \# of filters in the last conv layer of the generator, determines generator architecture
* ``--ndf``: \# of filters in the first conv layer of the discriminator, determines discriminator architecture
* ``--netD``: specify discriminator architecture \[ basic | n\_layers | pixel \]. The basic model is a 70x70 PatchGAN. n\_layers allows you to specify the layers in the discriminator. 
* ``--netG``: specify generator architecture \[ resnet\_9blocks | resnet\_6blocks | unet\_256 | unet\_128 | linearfilt \]
* ``--n_layers_D``: only used if netD==n\_layers
* ``--norm``: instance normalization or batch normalization \[ instance | batch | none \]
* ``--init_type``: network initialization \[ normal | xavier | kaiming | orthogonal \]
* ``--init_gain``: scaling factor for normal, xavier and orthogonal
* ``--no_dropout``: no dropout for the generator

Dataset parameters:
* ``--dataset_mode``: chooses how dataset loader \[ txm2sem | txm2sem3d | segmentation | image\_repair \]
* ``--direction``: AtoB or BtoA (where A is TXM and B is SEM, do not change)
* ``--serial_batches``: if flag is passed, takes images in order to make batches, otherwise takes them randomly
* ``--num_threads``: \# threads for loading data
* ``--batch_size``: input batch size
* ``--load_size``: scale images to this size (in data transform)
* ``--crop_size``: then crop images to this size (in data transform)
* ``--full_slice``: evaluate full image slices
* ``--max_dataset_size``: Maximum number of samples allowed per dataset. If the dataset directory contains more than max\_dataset\_size, only a subset is loaded.
* ``--preprocess``: scaling and cropping of images at load time \[ resize_and_crop | crop | scale\_width | scale\_width\_and\_crop | none \]
* ``--no_flip``: if specified, do not flip the images for data augmentation
* ``--display_winsize``: display window size for both visdom and HTML

Other parameters:
* ``--epoch``: which epoch to load (either epoch \# or set to ``latest`` to use latest cached model)
* ``--load_iter``: which iteration to load (if load\_iter > 0, the code will load models by ``--load_iter``; otherwise, the code will load models by ``--epoch``)
* ``--verbose``: if specified, print more debugging information
* ``--suffix``: customized suffix (name = name + suffix e.g. \{model\}\_\{netG\}\_size\{load_size\})

**Training-specific commands**

Display parameters (should not need to be altered):
* ``--display_freq``: frequency of showing training results on screen
* ``--display_ncols``: if positive, display all images in a single visdom web panel with certain number of images per row.
* ``--display_id``: window id of the web display
* ``--display_server``: visdom server of the web display
* ``--display_env``: visdom display environment name (default is "main")
* ``--display_port``: visdom port of the web display
* ``--update_html_freq``: frequency of saving training results to html
* ``--print_freq``: frequency of showing training results on console
* ``--no_html``: do not save intermediate training results to web checkpoint directory

Network saving and loading parameters:
* ``--save_latest_freq``: frequency of saving the latest results
* ``--save_epoch_freq``: frequency of saving checkpoints at the end of epochs
* ``--save_by_iter``: if flag passed, saves model by iteration
* ``--continue_train``: if flag passed, continue training by loading the latest model
* ``--epoch_count``: the starting epoch count, we save the model by <epoch\_count>, <epoch\_count>+<save\_latest\_freq>, ...
* ``--phase``: train, val, test, etc. Do not change this option to ensure proper behavior. 

Training parameters:
* ``--niter``: \# of iter at starting learning rate
* ``--niter_decay``: \# of iter to linearly decay learning rate to zero
* ``--beta1``: momentum term of adam
* ``--lr``: initial learning rate for adam
* ``--gan_mode``: the type of GAN objective. \[ vanilla| lsgan | wgangp \]
* ``--pool_size``: the size of image buffer that stores previously generated images
* ``--lr_policy``: learning rate policy. \[ linear | step | plateau | cosine \]
* ``--lr_decay_iters``: multiply by a gamma every ``lr_decay_iters`` iterations
* ``--weight_decay``: L2 regularization for the generator network

**Testing-specific commands**

* ``--ntest``: \# of test examples
* ``--results_dir``: saves results here (default ``./results/``)
* ``--aspect_ratio``: aspect ratio of result images
* ``--phase``: train, val, test, etc. Do not change this option to ensure proper behavior.
* ``--eval``: use eval mode during test time
* ``--num_test``: how many test images to run


## Relevant Publications
This code has been used in the following papers: 
- Anderson, T.I., Vega, B., Mckinzie, J., Wang, Y., Aryana, S., Kovscek, A.R. Three-Dimensional Source Rock Image Reconstruction through Multimodal Imaging. Nature Scientific Reports, *in review* 2021. 
- Anderson, T.I., Vega, B., Kovscek, A.R., 2020. [Multimodal imaging and machine learning to enhance microscope images of shale.](https://www.sciencedirect.com/science/article/pii/S0098300420305768?dgcid=rss_sd_all) Computers and Geosciences 145, 104593.

We have also presented this work at the following conferences:
- Timothy I Anderson. "Reconstruction and Synthesis of Source Rock Images at the Pore Scale." Paper to be presented at the SPE Annual Technical Conference and Exhibition, Dubai, UAE and Online, September 2021.
- Timothy I Anderson, Bolivia Vega, Laura Frouté, Kelly Guan, Anthony R Kovscek. International Conference on Machine Learning Tackling Climate Change with Machine Learning Workshop, “Improving Image-Based Characterization of Porous Media with Deep Generative Models.” Online, July 2021.
- Timothy I Anderson, Bolivia Vega, Laura Frouté, Kelly Guan, Anthony R Kovscek. International Conference on Learning Representations Workshop on Deep Learning for Simulation, “Simulation Domain Generation for Characterization of Shale Source Rocks.” Online, May 2021.
- Timothy Anderson, Bolivia Vega, Jesse Mckinzie, Yuhang Wang, Saman Aryana, Anthony Kovscek, American Geophysical Union Fall Meeting, “Three-Dimensional Source Rock Image Reconstruction through Multimodal Imaging.” Online, Dec. 2020.
- Timothy Anderson, Bolivia Vega, Anthony Kovscek, American Geophysical Union Fall Meeting, “Deep Learning and Multimodality Imaging to Improve Shale Fabric Characterization.” San Francisco, USA, Dec. 2019.

## Acknowledgements
This work was supported as part of the Center for Mechanistic Control of Unconventional Formations (CMC-UF), an Energy Frontier Research Center funded by the U.S. Department of Energy (DOE), Office of Science, Basic Energy Sciences (BES), under Award \# DE-SC0019165. Use of the Stanford Synchrotron Radiation Lightsource, SLAC National Accelerator Laboratory, is supported by the U.S. Department of Energy, Office of Science, Office of Basic Energy Sciences under Contract No. DE-AC02-76SF00515. Part of this work was performed at the Stanford Nano Shared Facilities (SNSF), supported by the National Science Foundation under award ECCS-1542152. Parts of this work were also supported by Total, the Siebel Scholars Foundation, and the SUPRI-A Industrial Affiliates. Thank you to Laura Froute, Cynthia Ross, Kelly Guan, Yijin Liu (SLAC), and Kevin Filter (Semion, LLC) for their help in developing this work. 

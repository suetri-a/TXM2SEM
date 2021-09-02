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

To begin, first install the dependencies listed here. The code requires the following packages:
- **PyTorch**: install from the [PyTorch website](https://pytorch.org/)
- **sci-kit image**:

### Data 

The framework implements a custom data loader

The data loader expects the TXM and SEM data to be stored in specific places within your working folder. This is how 


### Running the Model


### Command Line Interface Summary

General commands: 


Training-specific commands:


Testing-specific commands:


## Relevant Publications
This code has been used in the following papers: 
- Anderson, T.I., Vega, B., Mckinzie, J., Wang, Y., Aryana, S., Kovscek, A.R. Three-Dimensional Source Rock Image Reconstruction through Multimodal Imaging. Nature Scientific Reports, *in review* 2021. 
- Anderson, T.I., Vega, B., Kovscek, A.R., 2020. [Multimodal imaging and machine learning to enhance microscope images of shale.](https://www.sciencedirect.com/science/article/pii/S0098300420305768?dgcid=rss_sd_all) Computers and Geosciences 145, 104593.

We have also presented this work at the following conferences:
- Timothy I Anderson. "Reconstruction and Synthesis of Source Rock Images at the Pore Scale." Paper to be presented at the SPE Annual Technical Conference and Exhibition, Dubai, UAE and Online, September 2021.
- Timothy Anderson, Bolivia Vega, Jesse Mckinzie, Yuhang Wang, Saman Aryana, Anthony Kovscek, American Geophysical Union Fall Meeting, “Three-Dimensional Source Rock Image Reconstruction through Multimodal Imaging.” Online, Dec. 2020.
- Timothy Anderson, Bolivia Vega, Anthony Kovscek, American Geophysical Union Fall Meeting, “Deep Learning and Multimodality Imaging to Improve Shale Fabric Characterization.” San Francisco, USA, Dec. 2019.

## Acknowledgements
This work was supported as part of the Center for Mechanistic Control of Unconventional Formations (CMC-UF), an Energy Frontier Research Center funded by the U.S. Department of Energy (DOE), Office of Science, Basic Energy Sciences (BES), under Award \# DE-SC0019165. Use of the Stanford Synchrotron Radiation Lightsource, SLAC National Accelerator Laboratory, is supported by the U.S. Department of Energy, Office of Science, Office of Basic Energy Sciences under Contract No. DE-AC02-76SF00515. Part of this work was performed at the Stanford Nano Shared Facilities (SNSF), supported by the National Science Foundation under award ECCS-1542152. Parts of this work were also supported by Total, the Siebel Scholars Foundation, and the SUPRI-A Industrial Affiliates. Thank you to Laura Froute, Cynthia Ross, Kelly Guan, Yijin Liu (SLAC), and Kevin Filter (Semion, LLC) for their help in developing this work. 

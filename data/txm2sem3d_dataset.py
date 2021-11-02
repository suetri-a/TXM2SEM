"""Dataset class for creating 3D image stacks of translated TXM images

Will select 128 consecutive TXM images from [train-test]/txm_full_stack and translate 
    a 128*128*128 volume from these slices to SEM images. 
    
"""
from data.base_dataset import BaseDataset, get_transform
from util.util import mkdirs
# from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import glob
import re
import os
import random
from util import util
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


class Txm2sem3dDataset(BaseDataset):
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--patch_size', type=int, default=256, help='image patch size when performing subsampling')
        parser.add_argument('--save_name', type=str, default=None, help='directory to store the saved volume in the results folder')
        parser.add_argument('--x_ind', type=int, default=None, help='x-index for sampling image patches')
        parser.add_argument('--y_ind', type=int, default=None, help='y-index for sampling image patches')
        parser.add_argument('--z_ind', type=int, default=0, help='z-index for sampling image patches')
        
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)

        self.patch_size = opt.patch_size

        # set whether to perform downsampling and to include 
        if opt.model in ['srcnn', 'srgan']:
            self.downsample_factor = opt.downsample_factor
            if hasattr(opt, 'd_condition'):
                self.include_original_res = opt.d_condition
            else:
                self.include_original_res = False
        else:
            self.downsample_factor = None
        
        # Check that translation is not being performed during training
        assert not opt.isTrain, '3D translation not defined for training mode'

        set_inc = opt.save_name if opt.save_name is not None else opt.phase  # optionally allow for sampling from training set-connected TXM images
        self.save_txm_dir = os.path.join(opt.results_dir, opt.name, 'volume_txm_{}'.format(set_inc))  # save directory for TXM image patches
        self.save_pred_dir = os.path.join(opt.results_dir, opt.name, 'volume_pred_{}'.format(set_inc))  # save directory for translated images
        mkdirs([self.save_pred_dir, self.save_txm_dir])

        # Get image paths and discard all but the first N for N=opt.patch_size
        img_dir = './images/{}/txm_full_stack/'.format(opt.phase)  # original TXM image directory
        self.image_paths = sorted(glob.glob(img_dir+'*.tif'))
        if len(self.image_paths) > self.patch_size:
            self.image_paths = self.image_paths[:self.patch_size]

        # Define the default transform function from base transform funtion. 
        opt.no_flip = True
        self.transform = get_transform(opt)

        self.length = opt.patch_size
        self.patch_size = opt.patch_size

        # Get x-y indices for sampling 
        I_dummy = Image.open(self.image_paths[0])
        self.x_ind = np.random.randint(0, I_dummy.size[0]-opt.patch_size) if opt.x_ind is None else opt.x_ind
        self.y_ind = np.random.randint(0, I_dummy.size[0]-opt.patch_size) if opt.y_ind is None else opt.y_ind
        self.z_ind = opt.z_ind
        

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """

        try:
            txm_patch = Image.open(self.image_paths[index+self.z_ind]).crop((self.x_ind, self.y_ind, self.x_ind+self.patch_size, self.y_ind+self.patch_size))
        except:
            print("z-index {} invalid. Please enter a valid z-index value.".format(self.z_ind, str(index+self.z_ind).zfill(3) + '.png'))
            exit()

        data_A = self.transform(txm_patch)
        A_paths = os.path.join(self.save_txm_dir, str(index+self.z_ind).zfill(3) + '.png')
        txm_patch.save(A_paths)
        B_paths = os.path.join(self.save_pred_dir, str(index+self.z_ind).zfill(3) + '.png')
        data_B = torch.zeros_like(data_A)
        A_orig = torch.zeros_like(data_A)

        if self.downsample_factor is not None:
            data_A = F.interpolate(torch.unsqueeze(data_A,1), size=(int(self.patch_size/self.downsample_factor), int(self.patch_size/self.downsample_factor)))
            data_A = torch.unsqueeze(torch.squeeze(data_A), 0)

        # Data transformation needs to convert to tensor
        return {'A': data_A, 'B': data_B, 'A_orig': A_orig, 'A_paths': A_paths, 'B_paths': B_paths}


    def __len__(self):
        """Return the total number of images."""
        return self.length


    
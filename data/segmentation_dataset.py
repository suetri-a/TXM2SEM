"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import glob
import re
import os
import random
from util import util
import torch
import torchvision.transforms as transforms


class SegmentationDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--aligned', type=eval, default=True, help='optionally use aligned or unaligned image patches')
        parser.add_argument('--eval_mode', type=eval, default=False, help='determines whether dataset has fixed or random indices')
        parser.add_argument('--seg_mode', type=str, default='sem', help='input image type for segmentation [txm | semfake | sem]')
        parser.add_argument('--patch_size', type=int, default=256, help='image patch size when performing subsampling')
        parser.add_argument('--txm_dir', type=str, default='txm/', help='directory containing TXM images')
        parser.add_argument('--sem_dir', type=str, default='sem/', help='directory containing SEM images')
        parser.add_argument('--highdens_dir', type=str, default='highdensity/', help='directory containing high density region masks')
        parser.add_argument('--lowdens_dir', type=str, default='lowdensity/', help='directory containing high density region masks')
        parser.add_argument('--charge_dir', type=str, default='charge/', help='directory containing charge images')
        parser.add_argument('--num_train', type=int, default=10000, help='number of image patches to sample for training set')
        parser.add_argument('--num_test', type=int, default=1000, help='number of image patches to sample for test set')

        parser.set_defaults(max_dataset_size=10000, new_dataset_option=2.0)  # specify dataset-specific default values
        
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
        self.aligned = opt.aligned
        self.eval_mode = opt.eval_mode
        self.seg_mode = opt.seg_mode
        
        # get images for dataset;
        img_nums = []
        IMGS = []
        charges = []
        highdens = []
        lowdens = []
        
        if opt.isTrain:
            base_img_dir = './images/train/'
        else:
            base_img_dir = './images/test/'

        if self.sem_mode:
            source_dir = base_img_dir + opt.sem_dir
        else:
            source_dir = base_img_dir + opt.txm_dir
        
        charge_dir = base_img_dir + opt.charge_dir
        highdens_dir = base_img_dir + opt.highdens_dir
        lowdens_dir = base_img_dir + opt.lowdens_dir

        for f in glob.glob(source_dir+'*.tif'):
            img_nums.append(max(map(int, re.findall('\d+', f))))
            imnum = str(img_nums[-1]).zfill(3)
            
            if self.seg_mode=='sem':
                IMGS.append(np.asarray(Image.open(source_dir+imnum+'_SEM.tif').convert('L')))
            else:
                IMGS.append(np.asarray(Image.open(source_dir+imnum+'_TXM.tif').convert('L')))

            charges.append(np.asarray(Image.open(charge_dir+imnum+'_SEM_mask.tif')))
            highdens.append(np.asarray(Image.open(highdens_dir+'SEM_light'+imnum+'.tif')))
            lowdens.append(np.asarray(Image.open(lowdens_dir+'SEM_dark'+imnum+'.tif')))

        
        # Sort according to slice number
        sort_inds = np.argsort(img_nums)
        self.imgs = [IMGS[i] for i in sort_inds]
        self.charges = [charges[i] for i in sort_inds]
        self.lowdensity = [lowdens[i] for i in sort_inds]
        self.highdensity = [highdens[i] for i in sort_inds]

        # Define the default transform function from base transform funtion. 
        if opt.model in ['feedforward']:
            self.transform = get_transform(opt, convert=False)
        else:
            self.transform = get_transform(opt)

        # Get patch indices and save subset of patches
        if self.seg_mode=='sem':
            self.source_save_dir = os.path.join(opt.checkpoints_dir, opt.name, 'sample_imgs', 'sem')
        elif self.seg_mode=='semfake':
            self.source_save_dir = os.path.join(opt.checkpoints_dir, opt.name, 'sample_imgs', 'sem_fake')
        else:
            self.source_save_dir = os.path.join(opt.checkpoints_dir, opt.name, 'sample_imgs', 'txm')
        self.seg_save_dir = os.path.join(opt.checkpoints_dir, opt.name, 'sample_imgs', 'segmented')
        util.mkdirs([self.source_save_dir, self.highdensity_save_dir, self.lowdensity_save_dir])

        if opt.isTrain:
            self.length = opt.num_train
        else:
            self.length = opt.num_test

        # Sample fixed patch indices if set to evaluation mode
        if self.eval_mode:
            np.random.seed(999)
            random.seed(999)
            self.indices = []
            for i in range(self.length):
                inds_temp = self.get_aligned_patch_inds()
                self.indices.append(inds_temp)

                source_img, class_mask = self.get_patch(i)
                source_img, class_mask = util.tensor2im(torch.unsqueeze(source_img,0)), util.tensor2im(torch.unsqueeze(class_mask,0))

                source_img_path = self.source_save_dir + str(i).zfill(3) + '.png'
                class_path = self.seg_save_dir + str(i).zfill(3) + '.png'
                
                util.save_image(source_img, source_img_path)
                util.save_image(class_mask, class_path)
        

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

        data_A, data_B = self.get_patch(index)
        
        A_paths = self.source_save_dir + str(index).zfill(3) + '.png' 
        B_paths = self.seg_save_dir + str(index).zfill(3) + '.png'

        # Data transformation needs to convert to tensor
        return {'A': data_A, 'B': data_B, 'A_paths': A_paths, 'B_paths': B_paths}


    def __len__(self):
        """Return the total number of images."""
        return self.length


    def get_patch(self, index):
        '''
        Randomly sample patch from image stack
        ''' 
        if self.eval_mode:
            xcoord, ycoord, zcoord = self.indices[index] # Unpack indices
        else:
            indstemp = self.get_aligned_patch_inds()
            xcoord, ycoord, zcoord = indstemp
        
        # fix for performing same transform taken from: https://github.com/pytorch/vision/issues/9 
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img transforms
        img_patch = transforms.ToTensor()(Image.fromarray(self.imgs[zcoord][xcoord:xcoord+self.patch_size, ycoord:ycoord+self.patch_size]))
        highdens_patch = transforms.ToTensor()(Image.fromarray(self.highdensity[zcoord][xcoord:xcoord+self.patch_size, ycoord:ycoord+self.patch_size]))
        lowdens_patch = transforms.ToTensor()(Image.fromarray(self.lowdensity[zcoord][xcoord:xcoord+self.patch_size, ycoord:ycoord+self.patch_size]))

        # Create background class - 0, low density - 1, high density - 2
        classes_out = torch.zeros_like(img_patch) + lowdens_patch + 2*highdens_patch
        
        return img_patch, classes_out


    def get_aligned_patch_inds(self):
        
        W, H = self.charges[0].shape
        good_patch = False
        
        while not good_patch:
            # Sample random coordinate
            xcoord = np.random.randint(0, high = W-self.patch_size)
            ycoord = np.random.randint(0, high = H-self.patch_size)
            zcoord = np.random.randint(0, high = len(self.charges))
            
            # Extract image patches from random coordinate
            img_patch = self.imgs[zcoord][xcoord:xcoord+self.patch_size, ycoord:ycoord+self.patch_size]
            charge_patch = self.charges[zcoord][xcoord:xcoord+self.patch_size, ycoord:ycoord+self.patch_size]
            highdens_patch = self.highdensity[zcoord][xcoord:xcoord+self.patch_size, ycoord:ycoord+self.patch_size] 
            lowdens_patch = self.lowdensity[zcoord][xcoord:xcoord+self.patch_size, ycoord:ycoord+self.patch_size]  

            # Calculate number of pixels that are zero and are uncharged
            img_prop = np.sum(img_patch) / img_patch.size
            charge_prop = np.sum(charge_patch) / charge_patch.size
            highdens_prop =  np.sum(highdens_patch) / highdens_patch.size
            lowdens_prop =  np.sum(lowdens_patch) / lowdens_patch.size
            
            # Check if patch is acceptable, if not resample
            if (highdens_prop > 0.1 or lowdens_prop > 0.1) and img_prop > 0.75 and charge_prop > 0.95:
                good_patch = True

        return (xcoord, ycoord, zcoord)


    
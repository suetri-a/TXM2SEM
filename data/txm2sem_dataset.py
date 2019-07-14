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


class Txm2semDataset(BaseDataset):
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
        parser.add_argument('--patch_size', type=int, default=256, help='image patch size when performing subsampling')
        parser.add_argument('--txm_dir', type=str, default='txm/', help='directory containing TXM images')
        parser.add_argument('--sem_dir', type=str, default='sem/', help='directory containing SEM images')
        parser.add_argument('--charge_dir', type=str, default='charge/', help='directory containing TXM images')
        parser.add_argument('--num_train', type=int, default=10000, help='number of image patches to sample for training set')
        parser.add_argument('--num_test', type=int, default=1000, help='number of image patches to sample for test set')

        parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        
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
        self.aligned = opt.dataset_mode != 'unaligned'
        # get images for dataset;
        img_nums = []
        TXM = []
        SEM = []
        charges = []

        
        if opt.training:
            base_img_dir = '../images/train/'
        else:
            base_img_dir = '../images/test/'

        txm_dir = base_img_dir + opt.txm_dir
        sem_dir = base_img_dir + opt.sem_dir
        charge_dir = base_img_dir + opt.charge_dir

        for f in glob.glob(opt.txm_dir+'/*.tif'):
            imnum = f[6:9]
            img_nums.append(int(imnum))
            TXM.append(Image.open(txm_dir+imnum+'_TXM.tif'))
            SEM.append(Image.open(sem_dir+imnum+'_SEM.tif'))
            charges.append(Image.open(charge_dir+imnum+'_SEM_mask.tif'))
        
        # Sort according to slice number
        sort_inds = np.argsort(img_nums)
        self.txm = [TXM[i] for i in sort_inds]
        self.sem = [SEM[i] for i in sort_inds]
        self.charges = [charges[i] for i in sort_inds]

        # Get patch indices
        self.indices = []
        np.random.seed(999)

        if opt.training:
            num_patches = opt.num_train
        else:
            num_patches = opt.num_test

        for _ in range(num_patches):
            if self.aligned:
                inds_temp = self.get_aligned_patch_inds()
                self.indices.append(inds_temp)
            else:
                inds1_temp, inds2_temp = self.get_unaligned_patch_inds()
                self.indices.append([inds1_temp, inds2_temp])
        
        # define the default transform function from base transform funtion. 
        self.transform = get_transform(opt)

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

        path = None 
        
        if self.aligned:
            data_A, data_B = self.get_patch(self.indices[index])
        else:
            data_A, _ = self.get_patch(self.indices[index][0])
            _, data_B = self.get_patch(self.indices[index][1])

        # Data transformation needs to convert to tensor
        return {'data_A': data_A, 'data_B': data_B, 'path': path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.indices)


    def get_aligned_patch_inds(self):

        win = np.ceil(self.patch_size/2).astype(int)
        _, H, W = self.txm[0].shape
        good_patch = False
        
        while not good_patch:
            # Sample random coordinate
            xcoord = np.random.randint(win, high = H-win)
            ycoord = np.random.randint(win, high = W-win)
            zcoord = np.random.randint(0, high = len(self.txm))
            
            # Extract image patches from random coordinate
            sem_patch = self.sem[zcoord][:, xcoord - win:xcoord + win, ycoord - win:ycoord + win]
            txm_patch = self.txm[zcoord][:, xcoord - win:xcoord + win, ycoord - win:ycoord + win]
            charge_patch = np.greater(self.charges[zcoord][0, xcoord - win:xcoord + win, ycoord - win:ycoord + win], 0)     
            
            # Calculate mask of all black pixels across all three images
            mask = np.greater(charge_patch*sem_patch*txm_patch,0)

            # Calculate number of pixels that are zero and are uncharged
            mask_prop =  np.sum(mask) / sem_patch.size
            uncharge_prop =  np.sum(charge_patch) / sem_patch.size
            
            # Check if patch is acceptable, if not resample
            if uncharge_prop > 0.95 and mask_prop > 0.75:
                good_patch = True
        return (xcoord, ycoord, zcoord)

    def get_unaligned_patch_inds(self):
        inds1 = self.get_aligned_patch_inds()
        inds2 = self.get_aligned_patch_inds()
        return inds1, inds2

    def get_patch(self, inds):
        '''
        Randomly sample patch from image stack
        '''
        win = np.ceil(self.patch_size/2).astype(int)
        xcoord, ycoord, zcoord = inds # Unpack indices
        sem_patch = self.sem[zcoord][:, xcoord - win:xcoord + win, ycoord - win:ycoord + win]
        txm_patch = self.txm[zcoord][:, xcoord - win:xcoord + win, ycoord - win:ycoord + win]
        
        return txm_patch, sem_patch
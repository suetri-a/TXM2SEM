import torch
from .pix2pix_model import Pix2PixModel
from . import networks


class RepairModel(Pix2PixModel):
    """ 
    This is a small wrapper class around the pix2pix model that implements the SEM image repair models. 
        All this class does is change some of the default inputs for pix2pix This is done for convenience when running the numerical experiments

    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        Change pix2pix default options to match for the image repair task. 

        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)

        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='image_repair', input_nc=3)
        parser.add_argument('--sem_only', type=eval, default=False, help='use only SEM with charged patches as input')
        
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for L1 loss')
            

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        Pix2PixModel.__init__(self, opt)


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        sem_damaged = self.real_A[:,1,...]
        charge = torch.gt(self.real_A[:,2,...],0).float()
        output = self.netG(self.real_A)  # G(A)
        self.fake_B = charge*output + (1-charge)*sem_damaged
"""Model class FeedforwardModel

Simple image-to-image translation baseline based on regression loss.

Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1

You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
from jacobian import JacobianReg
from .base_model import BaseModel
from . import networks


class FeedforwardModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.set_defaults(dataset_mode='txm2sem')  # Use aligned data for this model
        if is_train:
            parser.add_argument('--lambda_regression', type=float, default=1.0, help='weight for the regression loss')  
            parser.add_argument('--regression_loss', type=str, default='L1', help='loss type for the regression (L2 or L1)') 
            parser.add_argument('--lambda_img_grad', type=float, default=0.0, help='weight to image gradient penalty')
        parser.set_defaults(norm='batch')
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['G']
        
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['data_A', 'data_B', 'fake_B']
        
        self.model_names = ['G'] # Models you want to save to disk.
        
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, gpu_ids=self.gpu_ids)
        
        if self.isTrain:  # only defined during training time
            # Loss function 
            if opt.regression_loss == 'L1':
                self.criterionLoss = torch.nn.L1Loss()
            elif opt.regression_loss == 'L2':
                self.criterionLoss = torch.nn.MSELoss()
            else:
                raise NotImplementedError('Loss type {} not implemented.'.format(opt.regression_loss))
            
            # Define optimizer
            self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]

            # initialize losses and criterions for image gradient penalty
            if opt.lambda_img_grad > 0.0:
                self.loss_names += ['G_img_grad', 'G_img']
                self.image_grad_reg = JacobianReg() # Jacobian regularization

        if self.isTrain: # define constants
            self.lambda_img_grad = opt.lambda_img_grad
        else:
            self.lambda_img_grad = 0.0

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        AtoB = self.opt.direction == 'AtoB'  # use <direction> to swap data_A and data_B
        self.data_A = input['A' if AtoB else 'B'].to(self.device)  # get image data A
        if self.lambda_img_grad > 0.0:
            self.data_A.requires_grad_(True)
        self.data_B = input['B' if AtoB else 'A'].to(self.device)  # get image data B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']  # get image paths

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.data_A)  # generate output image given the input data_A

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.fake_B has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_G_img = self.criterionLoss(self.fake_B, self.data_B) * self.opt.lambda_regression
        # self.loss_G_img_grad, _ = networks.cal_image_grad_penalty(self.fake_B, self.data_A, 
        #                                                         self.device, lambda_gp=self.lambda_img_grad)
        if self.lambda_img_grad > 0.0:
            fake_B_in = torch.reshape(self.fake_B, (self.fake_B.shape[0],-1))
            self.loss_G_img_grad = self.lambda_img_grad * self.image_grad_reg(self.data_A, fake_B_in)
        else:
            self.loss_G_img_grad = 0.0
        self.loss_G = self.loss_G_img + self.loss_G_img_grad
        self.loss_G.backward()       # calculate gradients of network G w.r.t. loss_G

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results
        self.optimizer.zero_grad()   # clear network G's existing gradients
        self.backward()              # calculate gradients for network G
        self.optimizer.step()        # update gradients for network G

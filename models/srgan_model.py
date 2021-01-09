import torch
from jacobian import JacobianReg
from .base_model import BaseModel
from . import networks


class SRGANModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=1000.0, help='weight for L1 loss')
            parser.add_argument('--lambda_img_grad', type=float, default=0.0, help='weight to image gradient penalty')
            parser.add_argument('--lambda_gp', type=float, default=0.0, help='weight for gradient penalty in wgangp loss')
        
        parser.add_argument('--d_condition', action="store_true", help='pass original resolution image into discriminator with generated image')
        parser.set_defaults(ngf=256, norm='batch')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A_orig', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, ds_fac=opt.downsample_factor)

        if self.isTrain:  # define a discriminator
            channels_in = opt.output_nc + opt.input_nc if opt.d_condition else opt.output_nc
            self.netD = networks.define_D(channels_in, opt.ndf, opt.netD, 
                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.d_condition = opt.d_condition
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # initialize losses and criterions for image gradient penalty
            if opt.lambda_img_grad > 0.0:
                self.loss_names += ['G_img_grad']
                self.image_grad_reg = JacobianReg() # Jacobian regularization
            if opt.lambda_gp > 0.0:
                self.loss_names += ['D_gp']

        if self.isTrain: # define constants
            self.lambda_img_grad = opt.lambda_img_grad
            self.lambda_gp = opt.lambda_gp
        else:
            self.lambda_img_grad = 0.0
            self.lambda_gp = 0.0

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        if self.lambda_img_grad > 0.0:
            self.real_A.requires_grad_(True)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_A_orig = input['A_orig']
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # d_in_fake = torch.cat((self.real_A_orig, self.fake_B), 1) if self.d_condition else self.fake_B
        fake_B = self.fake_B
        pred_fake = self.netD(fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_B = self.real_B
        pred_real = self.netD(real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # Gradient penalty
        self.loss_D_gp, _ = networks.cal_gradient_penalty(self.netD, real_B, fake_B.detach(), self.device, lambda_gp=self.lambda_gp)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 + self.loss_D_gp
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        pred_fake = self.netD(self.fake_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # Third, grad G(A)
        # self.loss_G_img_grad, _ = networks.cal_image_grad_penalty(self.fake_B, self.real_A, 
        #                                                         self.device, lambda_gp=self.lambda_img_grad)
        if self.lambda_img_grad > 0.0:
            fake_B_in = torch.reshape(self.fake_B, (self.fake_B.shape[0],-1))
            self.loss_G_img_grad = self.lambda_img_grad * self.image_grad_reg(self.real_A, fake_B_in)
        else:
            self.loss_G_img_grad = 0.0
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_img_grad
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

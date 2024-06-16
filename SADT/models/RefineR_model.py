import torch
from collections import OrderedDict
import time
import numpy as np
import os
import torch.nn.functional as F
import torch.nn as nn
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import util.util as util
from .distangle_model import DistangleModel
from PIL import ImageOps, Image
import cv2


def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
            # image_numpy = image_numpy.convert('L')
        #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return np.clip(image_numpy, 0, 255).astype(imtype)


class L_TV(nn.Module):
    def __init__(self):
        super(L_TV, self).__init__()

    def forward(self, x):
        _, _, h, w = x.size()
        count_h = (h - 1) * w
        count_w = (w - 1) * h

        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w - 1], 2).sum()
        return (h_tv / count_h + w_tv / count_w) / 2.0


class GradientLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(GradientLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                             f'Supported ones are: {_reduction_modes}')

    def forward(self, pred, target):
        _, cin, _, _ = pred.shape
        _, cout, _, _ = target.shape
        assert cin == 3 and cout == 3
        kx = torch.Tensor([[1, 0, -1], [2, 0, -2],
                           [1, 0, -1]]).view(1, 1, 3, 3).to(target)
        ky = torch.Tensor([[1, 2, 1], [0, 0, 0],
                           [-1, -2, -1]]).view(1, 1, 3, 3).to(target)
        kx = kx.repeat((cout, 1, 1, 1))
        ky = ky.repeat((cout, 1, 1, 1))

        pred_grad_x = F.conv2d(pred, kx, padding=1, groups=3)
        pred_grad_y = F.conv2d(pred, ky, padding=1, groups=3)
        target_grad_x = F.conv2d(target, kx, padding=1, groups=3)
        target_grad_y = F.conv2d(target, ky, padding=1, groups=3)

        loss = (
            nn.L1Loss(reduction=self.reduction)
            (pred_grad_x, target_grad_x) +
            nn.L1Loss(reduction=self.reduction)
            (pred_grad_y, target_grad_y))
        return loss * self.loss_weight


class PoissonGradientLoss(nn.Module):
    def __init__(self, reduction='mean'):
        """L_{grad} = \frac{1}{2hw}\sum_{m=1}^{H}\sum_{n=1}{W}(\partial f(I_{Blend}) - 
                       (\partial f(I_{Source}) + \partial f(I_{Target})))_{mn}^2
           See **Deep Image Blending** for detail.
        """
        super(PoissonGradientLoss, self).__init__()
        self.reduction = reduction

    def forward(self, source, target, blend, mask):
        f = torch.Tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).view(1, 1, 3, 3).to(target)
        f = f.repeat((3, 1, 1, 1))
        grad_s = F.conv2d(source, f, padding=1, groups=3) * mask
        grad_t = F.conv2d(target, f, padding=1, groups=3) * (1 - mask)
        grad_b = F.conv2d(blend, f, padding=1, groups=3)
        return nn.MSELoss(reduction=self.reduction)(grad_b, (grad_t + grad_s))

class PenumbraLoss(nn.Module):
    def __init__(self, reduction='sum'):
        """L_{grad} = \frac{1}{2hw}\sum_{m=1}^{H}\sum_{n=1}{W}(\partial f(I_{Blend}) - 
                       (\partial f(I_{Source}) + \partial f(I_{Target})))_{mn}^2

           See **Deep Image Blending** for detail.
        """
        super(PenumbraLoss, self).__init__()
        self.reduction = reduction

    def forward(self, E_mask, source, target, erode, dilate):
        mask = dilate - erode
        fsource1 = source * mask * E_mask
        ftarget1 = target * mask * E_mask
        fsource2 = source * mask * (1 - E_mask)
        ftarget2 = target * mask * (1 - E_mask)
        return nn.L1Loss(reduction=self.reduction)(fsource1, ftarget1) + 2.0 * nn.L1Loss(reduction=self.reduction)(fsource2, ftarget2)#[mask>0]


class RefineRModel(DistangleModel):
    def name(self):
        return 'auto exposure cvpr21'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(dataset_mode='expo_param')
        parser.add_argument('--wdataroot', default='None',
                            help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--use_our_mask', action='store_true')
        parser.add_argument('--mask_train', type=str, default=None)
        parser.add_argument('--mask_test', type=str, default=None)
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.loss_names = ['G_param', 'alpha', 'rescontruction']
        self.visual_names = ['input_img', 'out', 'final', 'outgt']
        self.model_names = ['G']
        # load/define networks
        opt.output_nc = 1

        self.ks = ks = opt.ks
        self.rks = opt.rks
        self.n = n = opt.n
        self.shadow_loss = opt.shadow_loss
        self.tv_loss = opt.tv_loss
        self.grad_loss = opt.grad_loss
        self.pgrad_loss = opt.pgrad_loss
        self.penum_loss = opt.penum_loss

        self.netG = networks.define_D(512, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netG.to(self.device)
        
        print(self.netG)
        
        if self.isTrain:
            # define loss functions
            self.MSELoss = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.bce = torch.nn.BCEWithLogitsLoss()
            # initialize optimizers
            self.optimizers = []

            if opt.optimizer == 'adam':
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            elif opt.optimizer == 'sgd':
                self.optimizer_G = torch.optim.SGD(self.netG.parameters(), momentum=0.9,
                                                   lr=opt.lr, weight_decay=1e-5)
            else:
                assert False

            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        self.imname = input['imname']
        #self.shadow_mask = (self.shadow_mask > 0.9).type(torch.float)  # * 2 - 1
        
        self.shadowfree_img = input['C'].to(self.device)
        self.shadow_color = input['FC'].to(self.device)
        self.shadow_L = input['FL'].to(self.device)
        self.shadowfree_color = input['GC'].to(self.device)
        self.shadowfree_L = input['GL'].to(self.device)        
        self.shadow_mask_dilate = input['B_dilate'].to(self.device)       
        self.shadow_mask_erode = input['B_erode'].to(self.device)        
        self.E_mask = input['E_mask'].to(self.device)
        
    def forward(self):
        inputG = torch.cat([self.input_img, self.shadow_L, self.shadow_mask], 1)
        inputM = self.netG(inputG)

        b, c, h, w = self.input_img.shape
        #self.input_img   self.input_img, 
        #self.final = inputM * self.shadow_mask_dilate + self.shadow_L * (1 - self.shadow_mask_dilate)
        self.final = self.shadow_L * (1 + 4 * inputM)

    def backward(self):
        # criterion = self.criterionL1
        lambda_ = self.opt.lambda_L1

        if self.tv_loss > 0:
            tv_loss = L_TV()(self.final - self.shadowfree_img) * lambda_ * self.tv_loss
        else:
            tv_loss = 0.0

        if self.grad_loss > 0:
            grad_loss = GradientLoss()(self.final, self.shadowfree_img) * lambda_ * self.grad_loss
        else:
            grad_loss = 0.0

        if self.pgrad_loss > 0:
            pgrad_loss = PoissonGradientLoss()(target=self.input_img, blend=self.final,
                                               source=self.shadowfree_img, mask=self.shadow_mask_dilate) \
                         * lambda_ * self.pgrad_loss
        else:
            pgrad_loss = 0.0

        if self.penum_loss > 0:
            penum_loss = PenumbraLoss()(E_mask = self.E_mask, target=self.shadowfree_L, source=self.final,
                                        erode=self.shadow_mask_erode, dilate=self.shadow_mask_dilate) \
                         * lambda_ * self.penum_loss
        else:
            penum_loss = 0.0

        self.loss_rescontruction = nn.L1Loss(reduction='sum')(self.final, self.shadowfree_L) * lambda_
        self.loss = self.loss_rescontruction + tv_loss + grad_loss + pgrad_loss + penum_loss
        self.loss.backward()

    def optimize_parameters(self):
        
        self.netG.zero_grad()
        self.zero_grad()
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward()
        self.optimizer_G.step()

    def zero_grad(self):
        self.netG.zero_grad()
        self.optimizer_G.zero_grad()

    def vis(self, e, s, path='', eval=False):
        if len(path) > 0:
            save_dir = os.path.join(self.save_dir, path)
        else:
            save_dir = self.save_dir
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        shadow = self.input_img
        
        output = self.final
        gt = self.shadowfree_img

        if not eval:
            #img = torch.cat([shadow, output, gt], dim=-1)[0, ...]
            img = output[0, ...]
            filename = os.path.join(save_dir, "epoch_%d_step_%d.png" % (e, s))

            img = tensor2im(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, img)

        else:
            filename = os.path.join(save_dir, self.imname[0])
            img = output[0, ...]
            img = tensor2im(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, img)

            #filename = os.path.join(save_dir, self.imname[0].replace('.png', '-o.png'))
            #img = ooutput[0, ...]
            #img = tensor2im(img)
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #cv2.imwrite(filename, img)

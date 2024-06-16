import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageChops
from PIL import ImageFilter
import torch
from pdb import set_trace as st
import random
import numpy as np
import time
import cv2


class ExpoParamDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        if 'train' in opt.phase:
            self.dir_A = os.path.join(opt.dataroot,'ISTD', opt.phase + 'JA')       
            self.dir_B = os.path.join(opt.dataroot,'ISTD', opt.phase + 'B')
            self.dir_C = os.path.join(opt.dataroot,'ISTD', opt.phase + 'JC')
            self.dir_FC = os.path.join(opt.dataroot,'ISTD', opt.phase + 'JFC')
            self.dir_FL = os.path.join(opt.dataroot,'ISTD', opt.phase + 'JFL')
            self.dir_GC = os.path.join(opt.dataroot,'ISTD', opt.phase + 'JGC')
            self.dir_GL = os.path.join(opt.dataroot,'ISTD', opt.phase + 'JGL')
        else:
            self.dir_A = os.path.join(opt.dataroot,'ISTD', opt.phase + 'JA')        
            self.dir_B = os.path.join(opt.dataroot,'ISTD', opt.phase + 'B')
            self.dir_C = os.path.join(opt.dataroot,'ISTD', opt.phase + 'JC')
            self.dir_FC = os.path.join(opt.dataroot,'ISTD', opt.phase + 'JFC')
            self.dir_FL = os.path.join(opt.dataroot,'ISTD', opt.phase + 'JFL')
            self.dir_GC = os.path.join(opt.dataroot,'ISTD', opt.phase + 'JGC')
            self.dir_GL = os.path.join(opt.dataroot,'ISTD', opt.phase + 'JGL')

        
        print(self.dir_A)
        self.A_paths, self.imname = make_dataset(self.dir_A)
        self.A_size = len(self.A_paths)
        self.B_size = self.A_size

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=opt.norm_mean,
                                               std=opt.norm_std)]

        self.transformA = transforms.Compose(transform_list)
        self.transformB = transforms.Compose([transforms.ToTensor()])

        if 'train' in opt.phase:
            self.is_train = True
        else:
            self.is_train = False

    def __getitem__(self, index):
        colet = {}
        coletC = {}
        A_path = self.A_paths[index % self.A_size]
        imname = self.imname[index % self.A_size]
        index_A = index % self.A_size

        B_path = os.path.join(self.dir_B, imname)#.replace('.jpg', '.png')
        if not os.path.isfile(B_path):
            B_path = os.path.join(self.dir_B, imname)
        A_img = Image.open(A_path).convert('RGB')

        ow = A_img.size[0]
        oh = A_img.size[1]
        w = np.float(A_img.size[0])
        h = np.float(A_img.size[1])
        if os.path.isfile(B_path):
            B_img = Image.open(B_path).convert('L')
        else:
            print('MASK NOT FOUND : %s' % (B_path))
            B_img = Image.fromarray(np.zeros((int(w), int(h)), dtype=np.float), mode='L')
                
        B_img_np = np.asarray(B_img)
        kernel = np.ones((7, 7), np.uint8)
        B_img_np_dilate = cv2.dilate(B_img_np, kernel, iterations=1)
        B_img_np_erode = cv2.erode(B_img_np, kernel, iterations=1)
        B_img_dilate = Image.fromarray(B_img_np_dilate)
        B_img_erode = Image.fromarray(B_img_np_erode)
        colet['B_dilate'] = B_img_dilate
        colet['B_erode'] = B_img_erode

        colet['C'] = Image.open(os.path.join(self.dir_C, imname)).convert('RGB')#.replace('.jpg', '.png')
        colet['FC'] = Image.open(os.path.join(self.dir_FC, imname)).convert('RGB')
        colet['FL'] = Image.open(os.path.join(self.dir_FL, imname)).convert('L')
        colet['GC'] = Image.open(os.path.join(self.dir_GC, imname)).convert('RGB')
        colet['GL'] = Image.open(os.path.join(self.dir_GL, imname)).convert('L')        
        
        E_mask = np.zeros((int(w), int(h)), dtype=np.float)
        w1 = int(w)//3
        h1 = int(h)//3
        for i in range (w1, w1*2):
            for j in range (h1, h1*2):
                E_mask[i,j] = 1

        colet['E_mask'] = Image.fromarray(E_mask, mode='L')
        
        loadSize = self.opt.loadSize
        if self.is_train and self.opt.randomSize:
            loadSize = np.random.randint(loadSize + 1, loadSize * 1.3, 1)[0]

        if self.opt.keep_ratio:
            if w > h:
                ratio = np.float(loadSize) / np.float(h)
                neww = np.int(w * ratio)
                newh = loadSize
            else:
                ratio = np.float(loadSize) / np.float(w)
                neww = loadSize
                newh = np.int(h * ratio)
        else:
            neww = loadSize
            newh = loadSize

        colet['A'] = A_img
        colet['B'] = B_img

        if self.is_train:
            t = [Image.FLIP_LEFT_RIGHT, Image.ROTATE_90]
            for i in range(0, 4):
                c = np.random.randint(0, 3, 1, dtype=np.int)[0]
                if c == 2: continue
                for i in ['A', 'B', 'C', 'FC', 'FL', 'GC', 'GL', 'B_dilate', 'B_erode', 'E_mask']:
                    if i in colet:
                        colet[i] = colet[i].transpose(t[c])

        if self.is_train:
            degree = np.random.randint(-20, 20, 1)[0]
            for i in ['A', 'B', 'C', 'FC', 'FL', 'GC', 'GL', 'B_dilate', 'B_erode', 'E_mask']:
                colet[i] = colet[i].rotate(degree)

        for k, im in colet.items():
            if self.is_train:
                colet[k] = im.resize((neww, newh), Image.NEAREST)
            else:
                colet[k] = im.resize((self.opt.fineSize, self.opt.fineSize), Image.NEAREST)

        w = colet['A'].size[0]
        h = colet['A'].size[1]

        for k, im in colet.items():
            colet[k] = self.transformB(im)

#        for i in ['A', 'C', 'B', 'B_dilate', 'B_erode', 'E_mask']:
#            if i in colet:
#                colet[i] = (colet[i] - 0.5) * 2

        if self.is_train:  # and not self.opt.no_crop:
            w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))
            for k, im in colet.items():
                colet[k] = im[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

        if self.is_train and (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(colet['A'].size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            for k, im in colet.items():
                colet[k] = im.index_select(2, idx)

        #for i in ['A', 'B', 'C', 'FC', 'FL', 'GC', 'GL', 'BI', 'B_dilate', 'B_erode', 'E_mask']:
        #    if i in colet:
        #        imgv = img_spilt(colet[i], 2, 2)
        #        for j in range (len(imgv)):
        #            na = i + str(j)
        #            colet[na] = imgv[j]

        #for i in ['A', 'B', 'C', 'FC', 'FL', 'GC', 'GL', 'BI', 'B_dilate', 'B_erode', 'E_mask']:
        #    if i in colet:
        #        imgv = img_spilt(colet[i], 2, 2)
        #        imgr = img_remake(colet[i], imgv, 2, 2)
        #        na = i + '_R'
        #        colet[na] = imgr

        for k, im in colet.items():
            colet[k] = im.type(torch.FloatTensor)
        
        colet['imname'] = imname
        colet['w'] = ow
        colet['h'] = oh
        colet['A_paths'] = A_path
        colet['B_baths'] = B_path

        return colet

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'ExpoParamDataset'

    def img_spilt(img, n, m):
        w, h = img.size
        h1 = h//n
        w1 = w//m
        cimage = []
        for i in range(h1, h + h1, h1):
            for j in range(w1, w + w1, w1):
                cimage.append(img.crop([j-w1, i-h1, j, i]))
        return cimage

    def img_remake(img, cimage, n, m):
        w, h = img.size
        h1 = h//n
        w1 = w//m
        img2 = Image.new(img.mode, (w, h))
        n = 0
        for i in range(h1, h + h1, h1):
            for j in range(w1, w + w1, w1):
                im = cimage[3 - n]
                n = n + 1
                img2.paste(im, box = [j-w1, i-h1, j, i])
        return img2

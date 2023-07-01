import os
import gzip
import torch
import random
import pickle
import numpy as np
from torch.utils import data
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from utils.data_from_list import data_fromlist, load_img
from utils.GaussianBlur import GaussianBlur
from utils.cutout import Cutout
#from RandAugment import RandAugment


import PIL
import PIL.ImageEnhance
from PIL import Image


PARAMETER_MAX = 10

def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)

def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX
    
    
def augment_pool():
    augs = [(Color, 0.9, 0.05)] #(AutoContrast, None, None)] #,
            #(Identity, None, None)] #,
            #(Brightness, 0.9, 0.05),
            #(Color, 0.9, 0.05),
            #(Contrast, 0.9, 0.05),
            #(Equalize, None, None),
            #(Identity, None, None),
            #(Posterize, 4, 4),
            #(Rotate, 30, 0),
            #(Sharpness, 0.9, 0.05),
            #(ShearX, 0.3, 0),
            #(ShearY, 0.3, 0),
            #(Solarize, 256, 0),
            #(TranslateX, 0.3, 0),
            #(TranslateY, 0.3, 0)]
    return augs


class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        return img


#########################################
# Office/Office-Home/DomainNet Datasets #
#########################################
class StrongWeakTransformations(object):
    def __init__(self, crop_size, n=1, m=10, s=1):
        #color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.weak = transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomCrop(crop_size),
                                transforms.RandomHorizontalFlip(),
                                GaussianBlur(kernel_size=15),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

        self.strong = transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomCrop(crop_size),
                                transforms.RandomHorizontalFlip(),
                                #transforms.RandomApply([color_jitter], p=0.8),
                                RandAugmentMC(n=n,m=m),
                                transforms.RandomGrayscale(p=0.2),
                                #Cutout(size=16),
                                GaussianBlur(kernel_size=15),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

        #self.strong.transforms.insert(0, RandAugment(n, m))

    def __call__(self, x):
        return self.weak(x), self.strong(x)


class GetData(data.Dataset):
    def __init__(self, args, img_paths, crop_size, aug=False):
        if aug:
            self.transforms = StrongWeakTransformations(crop_size, n=args.N, m=args.M)
        else:
            self.transforms =  transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(crop_size),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
        self.base_path = './data/{}'.format(args.domain)
        self.x, self.y = data_fromlist(img_paths)
        
    def __len__(self):
        """
        Number of images in the object dataset.
        """
        return self.x.shape[0]

    def __getitem__(self, index):
        """
        Return an item from the dataset.
        """
        x = self.x[index]
        x = load_img(os.path.join(self.base_path, x))
        x = self.transforms(x)
        y = self.y[index]
        return x, y


def get_dataset(args):
    txt_path = './data/txt/{}'.format(args.domain)

    txt_file_s = os.path.join(txt_path,
                     'labeled_source_images_{}.txt'.format(args.source))
    
    txt_file_t = os.path.join(txt_path,
                     'labeled_target_images_{}_{}.txt'.format(args.target, args.n_shots))
                              
    txt_file_val = os.path.join(txt_path,
                     'validation_target_images_{}_{}.txt'.format(args.target, args.n_val))
                              
    txt_file_unl = os.path.join(txt_path,
                     'unlabeled_target_images_{}_{}.txt'.format(args.target, args.n_val))

    if args.model_name == 'Alexnet':
        crop_size = 227
    else:
        crop_size = 224
    bs = args.batch_size

    source_data = GetData(args, txt_file_s, crop_size, aug=True)
    source_loader = torch.utils.data.DataLoader(source_data, pin_memory=True, num_workers=args.n_workers,
                    batch_size=min(bs//2, args.n_classes*args.n_shots), shuffle=True, drop_last=True)

    target_data = GetData(args, txt_file_t, crop_size, aug=True)
    target_loader = torch.utils.data.DataLoader(target_data, pin_memory=True, num_workers=args.n_workers,
                    batch_size=min(bs//2, args.n_classes*args.n_shots), shuffle=True, drop_last=True)

    target_data_val = GetData(args, txt_file_val, crop_size, aug=False)
    target_loader_val = torch.utils.data.DataLoader(target_data_val, pin_memory=True, 
                    num_workers=args.n_workers, batch_size=bs, shuffle=False, drop_last=False)

    target_data_unl = GetData(args, txt_file_unl, crop_size, aug=True)
    target_loader_unl = torch.utils.data.DataLoader(target_data_unl, pin_memory=True, num_workers=args.n_workers,
                        batch_size=bs, shuffle=True, drop_last=True)

    target_data_test = GetData(args, txt_file_unl, crop_size, aug=False)
    target_loader_test = torch.utils.data.DataLoader(target_data_test, pin_memory=True, 
                    num_workers=args.n_workers, batch_size=bs, shuffle=False, drop_last=False)

    return source_loader, target_loader, target_loader_unl, target_loader_val, target_loader_test

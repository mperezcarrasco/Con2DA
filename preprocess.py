import os
import torch
import gzip
import pickle
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from utils.data_from_list import data_fromlist, load_img
from utils.GaussianBlur import GaussianBlur
from utils.randaug import RandAugmentMC

#########################################
# Office/Office-Home/DomainNet Datasets #
#########################################
class StrongWeakTransformations(object):
    def __init__(self, crop_size, n=2, m=10):
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
                                RandAugmentMC(n=2, m=10),
                                GaussianBlur(kernel_size=15),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

        self.strong.transforms.insert(0, RandAugment(n, m))

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

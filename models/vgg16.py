import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

class VGG16(nn.Module):
    """ 
    VGG16 net pretrained on imagenet for Office dataset. 
    Same model as in the original PyTorch implementation.
    """
    def __init__(self, args):
        super(VGG16, self).__init__()

        model = models.vgg16(pretrained=args.pretrain)
        self.encoder = model.features

        self.head = nn.Sequential()
        for i in range(6):
            self.head.add_module("head" + str(i),
                                    model.classifier[i])
        self.head.add_module('final_head', nn.Linear(4096, args.out_dim))

    def forward(self, x):
        h = self.encoder(x)
        z = self.head(h.view(-1, 256 * 6 * 6))
        return z

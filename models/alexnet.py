import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

class Alexnet(nn.Module):
    """ 
    AlexNet pretrained on imagenet for Office dataset. 
    Same model as in the original PyTorch implementation.
    """
    def __init__(self, args):
        super(Alexnet, self).__init__()

        model = models.alexnet(pretrained=args.pretrain)
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

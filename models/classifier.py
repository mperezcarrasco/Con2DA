import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    """ 
    Linear classifier. 
    """
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(args.out_dim, args.n_classes, bias=False)
        self.temp = args.temperature

    def forward(self, x):
        x = F.normalize(x, dim=1)
        x_out = self.fc(x) / self.temp
        return x_out

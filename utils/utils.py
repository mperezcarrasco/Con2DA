import os
import json
import math
import torch

class EarlyStopping:
    """Early stopping as the convergence criterion.

        Args:
            args (string): hyperparameters for the training.
            patience (int): the model will stop if it not do improve in a patience number of epochs.

        Returns:
            stop (bool): if the model must stop.
            if_best (bool): if the model performance is better than the previous models.
    """
    def __init__(self, args):
        self.best_metric = 0.0
        self.counter = 0
        self.patience = args.patience
        self.directory = args.directory

    def count(self, metric, model, classifier):
        is_best = bool(metric > self.best_metric)
        self.best_metric = max(metric, self.best_metric)
        if is_best:
            self.counter = 0
            torch.save({'encoder': model.state_dict(), 
                        'classifier': classifier.state_dict()},
                        '{}/trained_parameters_classifier.pth'.format(self.directory))
        else:
            self.counter += 1
        if self.counter > self.patience:
            stop = True
        else:
            stop = False
        return stop, is_best

def save_metrics(metrics, root_dir, mode='val'):
    """save all the metrics."""
    mt_dir = os.path.join(root_dir, 'metrics_{}.json'.format(mode))
    with open(mt_dir, 'w') as mt:
        json.dump(metrics, mt)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

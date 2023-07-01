import os
import math
import torch
import argparse
import numpy as np
from sklearn.metrics import accuracy_score

from loss import NTXentLoss
from preprocess import get_dataset
from torch.utils.tensorboard import SummaryWriter
from models.main import build_network, build_classifier
from utils.utils import AverageMeter, EarlyStopping, save_metrics

from torch.optim import Adam
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR


def parse():
    """Parsing the arguments for the model"""
    parser = argparse.ArgumentParser('Arguments used for training stage')

    #Dataset
    parser.add_argument("--domain", type=str, default='multi',
                         choices=['office', 'office-home', 'multi'],
                         help="Domain from which the dataset belong.")
    parser.add_argument("--source", type=str, default='real',
                         choices=['webcam', 'amazon', 'dslr', 'real', 'sketch', 'painting', 'clipart', 'Real', 'Art', 'Product', 'Clipart'],
                         help="Dataset to be used as target for the experiment.")
    parser.add_argument("--target", type=str, default='sketch',
                         choices=['webcam', 'amazon', 'dslr', 'real', 'art', 'product', 'sketch', 'painting', 'clipart', 'Real', 'Art', 'Product', 'Clipart'],
                         help="Dataset to be used as target for the experiment.")
    parser.add_argument("--n_shots", type=int, default=3,
                         help="Number of labeled samples to be used for the target.")
    parser.add_argument("--n_val", type=int, default=3,
                         help="Number of labeled samples to be used for validation.")

    #Model
    parser.add_argument("--model_name", type=str, default='Alexnet',
                         choices=['Alexnet', 'VGG16', 'Resnet34'],
                         help="Name of the model to be used for the experiment.")
    parser.add_argument("--pretrain", type=bool, default=True,
                         help="If domain is not digits, if the model must be pretrained on ImageNet.")

    #Hyperparams
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Size of each mini-batch.")
    parser.add_argument("--lr", type=float, default=8e-05,
                        help="Learning rate for the Adam optimizer.")
    parser.add_argument("--patience", type=int, default=50,
                        help="Patience for the early stopping.")
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="Hyperparameter temperature.")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Hyperparameter threshold.")
    parser.add_argument("--N", type=int, default=1,
                        help="Hyperparameter N for RandAug.")
    parser.add_argument("--M", type=int, default=10,
                        help="Hyperparameter M for RandAug")
    
    #Others
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--n_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument("--num_iterations", type=int, default=5000,
                        help="Training epochs.")
    parser.add_argument("--report_every", type=int, default=50,
                        help="Number of iterations for the metrics to be reported.")
    
    args = parser.parse_args()
    
    parent_dir = 'Ablation/results'
    job_name = '{}_{}_{}_{}_{}shots_temp{}_lr{}_bs{}_n{}_m{}_prior_gray'.format(args.domain, args.source, args.target,
               args.model_name, args.n_shots, args.temperature, args.lr, args.batch_size, args.N, args.M)

    args.directory = os.path.join(parent_dir, job_name)

    # create dir to store the results.
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    return args


def print_and_log(writer, metrics, mode, epoch):
    """Printing and logging results every args.report_every number of iterations"""
    for metric, value in metrics.items():
        print("{}: {:.2f}".format(metric, value))
        writer.add_scalar('{}_{}'.format(metric, mode), value, epoch)
    print("##########################################")


def train(args,
        device,
        writer,
        model, 
        clf,
        dl_sup_s,
        dl_sup_t,
        dl_unsup_t,
        dl_val):
    """Semi-supervised domain adaptation trainer that conmbines the two powerful frameworks:
    
    1. SimCLR - A Simple Framework for Contrastive Learning of Visual Representations",
    Chen T., Kornblit S., Norouzi M., Hinton G.,
    ref: https://arxiv.org/abs/2002.05709

    2. "Supervised Contrastive Learning",
    Khosla P., Teterwark P., Wang C., Sarna A., Tian Y., et at. 
    ref: https://arxiv.org/abs/2004.11362

    Args:
        args (string): hyperparameters for the training.
        writer: to record the training.
        model (torch.nn.Module): model to be used for the pretraining stage.
        optimizer (torch.optim): optimizer to be used for the training.
        dl_sup_s (data.DataLoader): dataloader for the source labeled_data.
        dl_sup_t (data.DataLoader): dataloder for the target labeled data.
        dl_unsup_t (data.DataLoader): dataloder for the target unlabeled data.
    """
    contrastive = NTXentLoss(device, args.batch_size, args.temperature)

    earlystop = EarlyStopping(args)

    optimizer_f = Adam(model.parameters(), lr=args.lr, weight_decay=10e-6)
    scheduler_f = CosineAnnealingLR(optimizer_f, T_max=len(dl_unsup_t), eta_min=0, last_epoch=-1)
    
    optimizer_c = Adam(clf.parameters(), lr=args.lr*10, weight_decay=10e-6)
    scheduler_c = CosineAnnealingLR(optimizer_c, T_max=len(dl_unsup_t), eta_min=0, last_epoch=-1)

    stop=False
    for iteration in range(args.num_iterations):
        if iteration % len(dl_sup_s) == 0:
            iter_sup_s = iter(dl_sup_s)
        if iteration % len(dl_sup_t) == 0:
            iter_sup_t = iter(dl_sup_t)
        if iteration % len(dl_unsup_t) == 0:
            iter_unsup = iter(dl_unsup_t)

        #################################
        #  CrossEntropy Minimization    #
        #################################
        model.train()
        clf.train()
        optimizer_f.zero_grad()
        optimizer_c.zero_grad()

        (x_w_s, x_s_s), y_s  = next(iter_sup_s)
        (x_w_t, x_s_t), y_t  = next(iter_sup_t)

        x = torch.cat((x_w_s, x_s_s, x_w_t, x_s_t),dim=0).float().to(device)
        y = torch.cat((y_s,y_s,y_t,y_t),dim=0).long().to(device)

        #Compute loss

        z = model(x)
        pred = clf(z)
        l_xent = F.cross_entropy(pred, y)

        l_xent.backward(retain_graph=True)
        optimizer_f.step()
        optimizer_c.step()

        del x, z, x_w_s, x_s_s, x_w_t, x_s_t

        ###########################################
        #  Unsupervised Contrastive Minimization  #
        ###########################################
        clf.eval()
        optimizer_f.zero_grad()
        optimizer_c.zero_grad()

        (x_w_u, x_s_u), _  = next(iter_unsup)
        x_w_u, x_s_u = x_w_u.float().to(device), x_s_u.float().to(device)

        z_w = model(x_w_u)
        z_s = model(x_s_u)
        l_cont = contrastive(z_w, z_s)
        
        pred_w = clf(z_w)
        pred_s = clf(z_s)

        pseudo_label = torch.softmax(((pred_w+pred_s)/2).detach(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.gt(args.threshold).float()
        loss_unsup_w = (F.cross_entropy(pred_w, targets_u, reduction='none') * mask).mean()
        loss_unsup_s = (F.cross_entropy(pred_s, targets_u, reduction='none') * mask).mean()
        
        loss = l_cont + loss_unsup_w + loss_unsup_s

        loss.backward()

        optimizer_f.step()
       

        del x_w_u, x_s_u, z_w, z_s

        if iteration>150 and iteration%30==0:
            scheduler_f.step()
            scheduler_c.step()

        if iteration%args.report_every==0:
            pred = np.argmax(pred.detach().cpu().numpy(), axis=1)
            metrics = {'Accuracy': accuracy_score(y.cpu().numpy(), pred)*100,
                       'Total Loss': (l_xent + loss_unsup_s + loss_unsup_w).item()}
            print('=====> Training... iteration {}/5000'.format(iteration))
            print_and_log(writer, metrics, 'train', iteration)

            metrics = evaluate(model, clf, dl_val, device, writer=writer, iteration=iteration)
            #Early stopping checkpoint.
            stop, is_best = earlystop.count(metrics['Accuracy'], model, clf)
            if is_best:
                save_metrics(metrics, args.directory)
        if stop or iteration>=args.num_iterations:
            break

def evaluate(model, clf, dataloader, device, writer=None, iteration=-999):
    """Evaluation module"""
    accuracy = AverageMeter()
    loss = AverageMeter()

    model.eval()
    clf.eval()
    with torch.no_grad():
        for x, y in dataloader:
            # Computing the predicted class for each datapoint.
            x = x.float().to(device)
            z = model(x)
            pred = clf(z)

            y_pred = np.argmax(pred.cpu().numpy(), axis=1)
            accuracy.update(accuracy_score(y, y_pred)*100, pred.shape[0])
            loss.update(F.cross_entropy(pred, y.to(device)).item(), pred.shape[0])

    metrics = {'Accuracy': accuracy.avg,
               'Total Loss': loss.avg}
    if writer is not None:
        print('=====> Validating... iteration {}/5000'.format(iteration))
        print_and_log(writer, metrics, 'val', iteration)
    else:
        print('=====> Testing...')
        for metric, value in metrics.items():
            print("{}: {:.2f}".format(metric, value))
        print("##########################################")
    return metrics


def main():
    args = parse()

    #Create all the dataloaders.
    if args.domain == 'office':
        args.out_dim = 256
        args.n_classes=31
        dataloader_sup_s, dataloader_sup_t, dataloader_unsup, \
        dataloader_val, dataloader_test = get_dataset(args)
    #Create all the dataloaders.
    if args.domain == 'office-home':
        args.out_dim = 256
        args.n_classes=65
        dataloader_sup_s, dataloader_sup_t, dataloader_unsup, \
        dataloader_val, dataloader_test = get_dataset(args)
    elif args.domain == 'multi':
        args.out_dim = 256
        args.n_classes=126
        dataloader_sup_s, dataloader_sup_t, dataloader_unsup, \
        dataloader_val, dataloader_test = get_dataset(args)

    #If cuda is available on device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Tensorboard logger
    writer = SummaryWriter(args.directory)

    #Setting the model
    model = build_network(args).to(device)
    clf = build_classifier(args).to(device)

    # Training and testing the model.
    train(args, device, writer, model, clf, dataloader_sup_s, dataloader_sup_t, dataloader_unsup, dataloader_val)

    metrics = evaluate(model, clf, dataloader_test, device)
    save_metrics(metrics, args.directory, mode='test')


if __name__ == '__main__':
    main()

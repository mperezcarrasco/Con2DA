from models.alexnet import Alexnet
from models.vgg16 import VGG16
from models.resnet import ResNet34
from models.classifier import Classifier

import torch
import torch.backends.cudnn as cudnn

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def build_network(args):
    """Builds the feature extractor and the projection head.

        Args:
            args: Hyperparameters for the network building.

        Returns:
            model (torch.nn.Module): Network architecture.
    """
    # Checking if the network is implemented.
    implemented_networks = ('Alexnet', 'VGG16', 'Resnet34')
    assert args.model_name in implemented_networks

    model = None

    if args.model_name == 'Alexnet':
        model = Alexnet(args)

    elif args.model_name == 'VGG16':
        model = VGG16(args)

    elif args.model_name == 'Resnet34':
        model = ResNet34(args)

    # enable synchronized Batch Normalization
    if args.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        cudnn.benchmark = True
    return model


def build_classifier(args):
    """Builds the linear classifier.

        Args:
            args: Hyperparameters for the classifier building.

        Returns:
            classifier (torch.nn.Module): Network architecture.
    """
    # Checking if the network is implemented.
    implemented_networks = ('Alexnet', 'VGG16', 'Resnet34')
    assert args.model_name in implemented_networks

    return Classifier(args)

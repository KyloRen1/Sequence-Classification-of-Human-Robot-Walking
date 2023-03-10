from torch import nn
from .focal import FocalLoss

def create_loss_function(config):
    if config.experiment_kwargs.loss_function.name == 'cross_entropy':
        loss = nn.CrossEntropyLoss(
            label_smoothing = config.experiment_kwargs.loss_function.label_smoothing,
            reduction = config.experiment_kwargs.loss_function.reduction
        )
    elif config.experiment_kwargs.loss_function.name == 'focal_loss':
        loss = FocalLoss(
            label_smoothing = config.experiment_kwargs.loss_function.label_smoothing,
            gamma = config.experiment_kwargs.loss_function.gamma,
            reduction = config.experiment_kwargs.loss_function.reduction
        )
    else:
        raise NotImplementedError
    return loss
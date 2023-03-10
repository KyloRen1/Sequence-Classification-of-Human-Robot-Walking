import torch

def create_optimizer(config, model):
    if config.opt == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr = config.lr,
            weight_decay = config.weight_decay)
    else:
        raise NotImplementedError
    return optimizer

def create_scheduler(experiment_config, optimizer):

    if experiment_config.sched == 'cosine_annealing_restarts':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = 1,
            T_mult = 2,
            eta_min = experiment_config.min_lr) # Minimum learning rate.
    elif experiment_config.sched == 'cosine_annealing':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = experiment_config.iter_max, # Maximum number of iterations.
            eta_min = experiment_config.min_lr) # Minimum learning rate.
    else:
        raise NotImplementedError
    return lr_scheduler

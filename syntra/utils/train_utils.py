import math


def adjust_learning_rate_poly(base_lr, iters, max_iters, power=2):
    lr = base_lr * ((1 - float(iters) / max_iters) ** power)
    return lr


def adjust_learning_rate_multistep(base_lr, iters, max_iters):
    lr = base_lr * 0.5 ** ((iters + 1) // 10000)
    return lr


def adjust_learning_rate_cosine(optimizer, base_lr, epoch, total_epoch, warmup_epochs):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = base_lr * epoch / warmup_epochs
    else:
        lr = base_lr * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (total_epoch - warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_learning_rate_exp_decay(optimizer, base_lr, itr, max_itr, warmup_itrs):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if itr < warmup_itrs:
        lr = base_lr * itr / warmup_itrs
    else:
        lr = base_lr * (1. - (itr - warmup_itrs) / max_itr) ** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_learning_rate_constant(optimizer, base_lr, itr, warmup_itrs):
    """Warm up, then hold the learning rate flat.

    For the iteration-count calibration: under a decaying schedule every checkpoint sits at a
    different learning rate, so a validation curve over checkpoints measures the schedule as
    much as convergence. Held flat, the curve says where the run stops improving, which is the
    quantity being calibrated. Not for the final runs - those anneal.
    """
    lr = base_lr * itr / warmup_itrs if itr < warmup_itrs else base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



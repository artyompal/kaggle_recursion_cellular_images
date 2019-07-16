''' Learning rate schedulers. '''

import json
import math

from typing import Any, Tuple

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_sched

from torch.optim import Optimizer
from optimizers import set_lr


class CosineLRWithRestarts():
    ''' Decays learning rate with cosine annealing, normalizes weight decay
    hyperparameter value, implements restarts.
    https://arxiv.org/abs/1711.05101

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        batch_size: minibatch size
        epoch_size: training samples per epoch
        restart_period: epoch count in the first restart period
        period_inc: period increment value
        period_max: maximum period value, in epochs


    Example:
        >>> scheduler = CosineLRWithRestarts(optimizer, 32, 1024, restart_period=5, period_inc=1)
        >>> for epoch in range(100):
        >>>     scheduler.epoch_step()
        >>>     train(...)
        >>>         ...
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>         scheduler.step()
        >>>     validate(...)
    '''

    def __init__(self, optimizer, batch_size, epoch_size, restart_period=100,
                 period_inc=2, max_period=100, last_epoch=-1, # eta_threshold=1000,
                 verbose=False, min_lr=1e-7):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an"
                                   " optimizer".format(i))

        self.base_lrs = list(map(lambda group: group['initial_lr'],
                                 optimizer.param_groups))

        self.last_epoch = last_epoch
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        # self.eta_threshold = eta_threshold
        self.period_inc = period_inc
        self.max_period = max_period
        self.verbose = verbose
        self.base_weight_decays = list(map(lambda group: group['weight_decay'],
                                           optimizer.param_groups))
        self.restart_period = restart_period
        self.restarts = 0
        self.t_epoch = -1
        self.min_lr = min_lr

    # def _schedule_eta(self) -> Tuple[float, float]:
    #     ''' Threshold value could be adjusted to shrink eta_min and eta_max values. '''
    #     eta_min = 0
    #     eta_max = 1
    #     if self.restarts <= self.eta_threshold:
    #         return eta_min, eta_max
    #     else:
    #         d = self.restarts - self.eta_threshold
    #         k = d * 0.09
    #         return (eta_min + k, eta_max - k)

    def get_lr(self, t_cur: int) -> Any:
        eta_min, eta_max = 0, 1

        eta_t = (eta_min + 0.5 * (eta_max - eta_min)
                 * (1. + math.cos(math.pi *
                                  (t_cur / self.restart_period))))

        weight_decay_norm_multi = math.sqrt(self.batch_size /
                                            (self.epoch_size *
                                             self.restart_period))
        lrs = [base_lr * eta_t for base_lr in self.base_lrs]
        weight_decays = [base_weight_decay * eta_t * weight_decay_norm_multi
                         for base_weight_decay in self.base_weight_decays]

        return zip(lrs, weight_decays)

    def _set_batch_size(self) -> None:
        d, r = divmod(self.epoch_size, self.batch_size)
        batches_in_epoch = d + 2 if r > 0 else d + 1
        self.batch_increment = iter(np.linspace(0, 1, batches_in_epoch))

    def epoch_step(self) -> bool:
        ''' Returns true if we started new cosine anneal period this epoch. '''
        self.last_epoch += 1
        self.t_epoch += 1
        self._set_batch_size()
        return self.step()

    def step(self) -> bool:
        res = False
        t_cur = self.t_epoch + next(self.batch_increment)

        for param_group, (lr, weight_decay) in zip(self.optimizer.param_groups,
                                                   self.get_lr(t_cur)):
            param_group['lr'] = max(lr, self.min_lr)
            param_group['weight_decay'] = weight_decay

        if self.t_epoch % self.restart_period < self.t_epoch:
            res = True
            if self.verbose:
                print("restart at epoch {}".format(self.last_epoch))

            self.restart_period = min(self.restart_period + self.period_inc,
                                      self.max_period)
            self.restarts += 1
            self.t_epoch = 0

        return res


def step(optimizer, last_epoch, step_size=10, gamma=0.1, **_) -> Any:
    return lr_sched.StepLR(optimizer, step_size=step_size, gamma=gamma,
                           last_epoch=last_epoch)

def multi_step(optimizer, last_epoch, milestones=[500, 5000], gamma=0.1, **_) -> Any:
    if isinstance(milestones, str):
        milestones = json.loads(milestones)

    return lr_sched.MultiStepLR(optimizer, milestones=milestones, gamma=gamma,
                                last_epoch=last_epoch)

def exponential(optimizer, last_epoch, gamma=0.995, **_) -> Any:
    return lr_sched.ExponentialLR(optimizer, gamma=gamma, last_epoch=last_epoch)

def none(optimizer, last_epoch, **_) -> Any:
    return lr_sched.StepLR(optimizer, step_size=10000000, last_epoch=last_epoch)

def reduce_lr_on_plateau(optimizer, last_epoch, mode='max', factor=0.1,
                         patience=10, threshold=0.0001, threshold_mode='rel',
                         cooldown=0, min_lr=0, **_) -> Any:
    return lr_sched.ReduceLROnPlateau(optimizer, mode=mode, factor=factor,
                                      patience=patience, threshold=threshold,
                                      threshold_mode=threshold_mode,
                                      cooldown=cooldown, min_lr=min_lr)

def cyclic_lr(optimizer, last_epoch, base_lr=0.001, max_lr=0.01,
              epochs_up=1, epochs_down=None, epoch_size=None, mode='triangular',
              gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=False,
              base_momentum=0.8, max_momentum=0.9, **_) -> Any:
    def exp_range_scale_fn(x):
        res = gamma ** (x - 1)
        return res

    last_epoch = -1
    step_size_up = epochs_up * epoch_size
    step_size_down = step_size_up if epochs_down is None else epochs_down * epoch_size
    return lr_sched.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                             step_size_up=step_size_up, step_size_down=
                             step_size_down, mode=mode, scale_fn=exp_range_scale_fn,
                             scale_mode=scale_mode, cycle_momentum=
                             cycle_momentum, base_momentum=base_momentum,
                             max_momentum=max_momentum, last_epoch=last_epoch)

def cosine_lr(optimizer, last_epoch, batch_size=None, epoch_size=None,
              start_lr=1e-4, restart_period=100, period_inc=2, max_period=100,
              verbose=False, min_lr=1e-7):
    last_epoch = -1
    set_lr(optimizer, start_lr)
    return CosineLRWithRestarts(optimizer, batch_size, epoch_size * batch_size,
                                restart_period, period_inc, max_period,
                                last_epoch, verbose, min_lr)

def get_scheduler(config, optimizer, epoch_size, last_epoch=-1):
    func = globals().get(config.scheduler.name)
    return func(optimizer, last_epoch, epoch_size=epoch_size,
                batch_size=config.train.batch_size,
                start_lr=config.optimizer.params.lr,
                **config.scheduler.params)

def is_scheduler_continuous(scheduler) -> bool:
    return type(scheduler) in [lr_sched.ExponentialLR, lr_sched.CosineAnnealingLR,
                               lr_sched.CyclicLR, CosineLRWithRestarts]

def get_warmup_scheduler(config, optimizer, epoch_size) -> Any:
    return lr_sched.CyclicLR(optimizer, base_lr=0,
                             max_lr=config.optimizer.params.lr,
                             step_size_up=config.train.warmup.epochs * epoch_size,
                             step_size_down=0,
                             cycle_momentum=False,
                             mode='triangular')

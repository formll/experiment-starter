from loguru import logger
import torch
import numpy as np

from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from .averagers import ModelAverager
from ..config import Config


def setup_optimizer(args: Config, model: torch.nn.Module):
    args.base_lr = base_lr = (args.base_lr if isinstance(args.base_lr, float)
                              else eval(args.base_lr))
    logger.info(f'Base LR for optimization evaluated to {base_lr}')
    weights_list = [p for n, p in model.named_parameters() if p.requires_grad and 'bias' not in n]
    biases_list = [p for n, p in model.named_parameters() if p.requires_grad and 'bias' in n]
    param_list = [dict(params=weights_list, weight_decay=args.weight_decay),
                  dict(params=biases_list, weight_decay=args.weight_decay_biases)]

    if args.optim_alg == 'sgd':
        momentum = float(args.momentum)
        optimizer = SGD(param_list,
                        lr=base_lr,
                        momentum=momentum,
                        weight_decay=args.weight_decay,
                        nesterov=momentum > 0.0)
    elif args.optim_alg in ('adam', 'adamw', 'amsgrad', 'amsgradw'):
        betas = tuple(map(float, str(args.momentum).split(',')))
        if len(betas) == 1:
            betas = (betas[0], 0.999)
        optim_cls = AdamW if args.optim_alg.endswith('w') else Adam
        optimizer = optim_cls(param_list,
                              lr=base_lr,
                              betas=betas,
                              eps=args.ada_eps,
                              weight_decay=args.weight_decay,
                              amsgrad='amsgrad' in args.optim_alg)
    logger.info('Set up optimizer: %s' % optimizer)

    train_batches = args.train_length
    if args.train_length_unit == 'epoch':
        if args.steps_per_epoch is None:
            train_batches *= args.data_n_examples['train'] / args.batch_size_accum
        else:
            train_batches *= args.steps_per_epoch
    elif args.train_length_unit == 'sample':
        train_batches /= args.batch_size_accum
    else:  # args.train_length_unit == 'step':
        train_batches *= args.grad_accum_steps

    if args.lr_schedule == 'constant':
        lr_factor_func = lambda epoch: 1
    elif args.lr_schedule.startswith('poly'):
        if args.lr_schedule == 'poly':
            power = 0.5
        else:
            power = float(args.lr_schedule.split('_')[-1])
        lr_factor_func = lambda batch: (1 + batch) ** (-power)
    elif args.lr_schedule.startswith('step'):
        if args.lr_schedule == 'step':
            times, factor = 4, 0.1  # this will decrease the lr by 0.1, 4 times
        else:
            times, factor = args.lr_schedule.split('_')[1:]
            factor = float(factor)
            times = list(map(float, times.split(',')))
        if len(times) == 1:
            times = times[0]
            lr_factor_func = lambda batch: factor ** int(batch / train_batches *
                                                         (times + 1))
        else:
            milestones = np.cumsum(times) / np.sum(times) * train_batches
            lr_factor_func = lambda batch: factor ** (
                (batch >= milestones).sum())
    elif args.lr_schedule == 'cosine':
        lr_factor_func = lambda batch: 0.5 * (
                1 + np.cos(np.pi * batch / train_batches))
    else:
        raise ValueError('Unknown LR schedule %s' % args.lr_schedule)
    lr_scheduler = LambdaLR(optimizer, lr_factor_func)

    averager = ModelAverager(model, args.averaging)

    return optimizer, averager, lr_scheduler, train_batches
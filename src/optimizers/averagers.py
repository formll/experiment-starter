from loguru import logger
import numpy as np
import torch


from copy import deepcopy


class ModelAverager(object):
    def __init__(self, model, method: str):
        self.t = 1
        self.model = model

        method = method or 'none'
        method, *args = method.split('_')
        self.method = method

        if method == 'none':
            self.av_model = model
            return
        else:
            self.av_model = deepcopy(model)
        if method == 'poly':
            self.eta = 0.0 if not args else float(args[0])
        elif method == 'ema':
            self.gamma = 0.99 if not args else float(args[0])
        else:
            logger.error(f'Unknown averaging method {method}')

    def step(self):
        method = self.method
        if method == 'none':
            return
        
        t = self.t
        model_sd = self.model.state_dict()
        av_sd = self.av_model.state_dict()

        for k in model_sd.keys():
            if isinstance(av_sd[k], (torch.LongTensor, torch.cuda.LongTensor)):  # these are buffers that store how many batches batch norm has seen so far
                av_sd[k].copy_(model_sd[k])
                continue
            if method == 'poly':
                av_sd[k].mul_(1 - ((self.eta + 1) / (self.eta + t))).add_(
                    model_sd[k], alpha=(self.eta + 1) / (self.eta + t)
                )
            elif method == 'ema':
                av_sd[k].mul_(1-self.gamma).add_(model_sd[k], alpha=self.gamma)

        self.t += 1

    def reset(self):
        self.t = 2

    @property
    def averaged_model(self):
        return self.av_model

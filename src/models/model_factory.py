import torch
from loguru import logger
import numpy as np

from time import sleep

from torch.nn import DataParallel

from timm import create_model as create_timm_model
from .tiny_models import create_model as create_tiny_model
from .clip_models import create_model as create_clip_model
from .linear_models import create_model as create_linear_model

from ..config import Config


def setup_model(dataset, args: Config):
    args.model_output_dim = args.model_output_dim or args.data_n_classes
    if args.model_output_dim is None:
        logger.warning('Could not determine the size of the output; will resort to the model default value')
    shape_args = dict(num_classes=args.model_output_dim) if args.model_output_dim else {}

    if args.model.startswith('timm/'):
        create_model = create_timm_model
    elif args.model.startswith('tiny/'):
        create_model = create_tiny_model
    elif args.model.startswith('clip/'):
        create_model = create_clip_model
    elif args.model.startswith('linear'):
        if isinstance(args.model_input_shape, (tuple, list)):
            input_dim = args.model_input_shape[-1]
        elif isinstance(args.model_input_shape, int):
            input_dim = args.model_input_shape
        else:
            logger.info(f'Inferring linear model input shape from data')
            x, _ = next(iter(dataset))
            assert x.ndim == 1, ValueError('Cannot apply linear layer to multidimensional array data')
            args.model_input_shape = list(x.shape)
            input_dim = args.model_input_shape[-1]
            logger.info(f'Inferred linear model input dimension is {input_dim}')
        shape_args['input_dim'] = input_dim
        create_model = create_linear_model
    else:
        raise ValueError(f'Unsupported model {args.model}')

    with torch.random.fork_rng():
        if args.seed_model is not None:
            torch.manual_seed(args.seed_model)  # ?

        for _ in range(100):  # to avoid deadlocks and other weird stuff when loading the model
            try:
                model = create_model(args.model.split('/', 1)[-1],
                                     pretrained=args.model_pretrained,
                                     **shape_args)
                break
            except RuntimeError as e:
                logger.warning(f'failed to load model, error message: {e}')
                logger.info('trying again')
                sleep(1 + float(np.random.rand(1) * 5))

        model_head = model.get_classifier() if hasattr(model, 'get_classifier') else model

    args.model_output_dim = model_head.weight.shape[1]

    if args.model_head_init is None or args.model_head_init == 'rand':
        pass
    elif args.model_head_init == 'zero':
        model_head.weight.data.zero_()
        if hasattr(model_head, 'bias') and model_head.bias is not None:
            model_head.bias.data.zero_()

    logger.info(f'Model has {np.sum([p.numel() for p in model.parameters() if p.requires_grad])} '
                f'trainable parameters')

    model = model.to(args.device).eval()

    if args.multi_gpu and args.device == 'cuda':  # (big) todo: better multi-gpu (and multinode?) training with DistributedDataParallel
        model = DataParallel(model)
        if hasattr(model.module, 'default_cfg'):
            model.default_cfg = model.module.default_cfg

    model = torch.compile(model, mode="max-autotune")

    return model

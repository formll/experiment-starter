import pdb

import torch
from torch.utils.data import DataLoader, TensorDataset
from timm.data import create_dataset as create_timm_dataset
from timm.data import create_loader as create_timm_dataloader

from loguru import logger

from .libsvm import create_libsvm_dataset
from .chunked_iterable import ChunkedIterable

from src.config import Config, CLASSIFICATION_LOSSES


def setup_datasets(args: Config):
    assert args.batch_size % args.grad_accum_steps == 0, ValueError('Number of gradient accumulation steps must equally divide batch size')
    args.batch_size_accum = batch_size = args.batch_size // args.grad_accum_steps

    eval_batch_size = args.eval_batch_size or args.batch_size

    datasets = {}

    if isinstance(args.data_splits_map, str):
        args.data_splits_map = eval(args.data_splits_map)

    assert 'train' in args.data_splits_map and args.data_splits_map['train'], ValueError('Must have a valid specification for the train split')

    if args.dataset.startswith('timm/'):
        def create_dataset(split_spec, is_train):
            return create_timm_dataset(
                args.dataset.replace('timm/', ''), root=args.data_dir, split=split_spec, is_training=is_train,
                batch_size=batch_size if is_train else eval_batch_size, download=args.data_download)
    elif args.dataset.startswith('libsvm/'):
        if not args.data_download:
            logger.warning('libsvmdata always downloads the dataset if not present')

        def create_dataset(split_spec, is_train):
            return create_libsvm_dataset(
                args.dataset.replace('libsvm/', ''), root=args.data_dir, split=split_spec)
    else:
        raise ValueError(f'Unsupported dataset {args.dataset}')

    for split, split_spec in args.data_splits_map.items():
        if split_spec is None:
            continue
        datasets[split] = create_dataset(split_spec, split == 'train')

    # update bookkeeping in args
    args.data_n_examples = {split: len(dataset) for split, dataset in datasets.items()}
    if args.data_n_classes is None:
        if hasattr(datasets['train'], 'num_classes'):
            args.data_n_classes = datasets['train'].num_classes
        elif args.dataset.startswith('timm/tfds'):  # try to automatically infer number of classes
            args.data_n_classes = datasets['train'].reader.builder.info.features['label'].num_classes
        elif args.loss in CLASSIFICATION_LOSSES:
            logger.info('Inferring number of classes based on dataset')
            if isinstance(datasets['train'], TensorDataset):
                args.data_n_classes = int(max(datasets['train'].tensors[1].numpy()) + 1)
            else:
                args.data_n_classes = int(max(y for _, y in datasets['train']) + 1)
        else:
            logger.warning('Number of classes not provided and was not inferred from dataset')

    return datasets


def setup_dataloaders(datasets, model, args: Config):
    if args.model.split('/')[0] in ('timm', 'tiny', 'clip'):  # use timm dataloaders with compatible models
        # todo: allow fancier aug config
        if hasattr(datasets['train'], 'loading_config'):
            loading_config = datasets['train'].loading_config
        else:
            loading_config = dict(
                scale=(0.08, 1.0),         # aka "inception crop"
                ratio=(3. / 4., 4. / 3.),  # aka "inception crop"
                hflip=0.5,
                color_jitter=0.4,
            )
        if hasattr(model, 'default_cfg'):
            loading_config.update(
                {k: model.default_cfg[k] for k in ('input_size', 'interpolation', 'mean', 'std', 'crop_pct')}
            )

        if args.model_input_shape is not None:
            loading_config['input_size'] = tuple(args.model_input_shape)
        else:
            args.model_input_shape = list(loading_config['input_size'])

        def create_dataloder(dataset, is_train):
            return create_timm_dataloader(dataset,
                                          batch_size=args.batch_size_accum if is_train else args.eval_batch_size,
                                          is_training=is_train,
                                          num_workers=args.num_workers,
                                          device=torch.device(args.device),
                                          **loading_config
                                          )
    else:  # use vanilla dataloader
        def create_dataloder(dataset, is_train):
            return DataLoader(dataset,
                              batch_size=args.batch_size_accum if is_train else args.eval_batch_size,
                              shuffle=is_train,
                              drop_last=is_train
                              )

    loaders = {}
    for split, dataset in datasets.items():
        loaders[split] = create_dataloder(dataset, split == 'train')

    # setup chunking on training dataloader
    if args.samples_per_epoch is not None:
        assert args.steps_per_epoch is None, ValueError('Either samples_per_epoch or steps_per_epoch cannot must be None')
        args.steps_per_epoch = args.samples_per_epoch // args.batch_size
    if args.steps_per_epoch is not None:
        args.samples_per_epoch = args.steps_per_epoch * args.batch_size
        logger.info(f'Redefining epoch to be {args.steps_per_epoch} steps')
        loaders['train'] = ChunkedIterable(loaders['train'], args.steps_per_epoch * args.grad_accum_steps)
        loaders['train'].dataset = datasets['train']

    return loaders


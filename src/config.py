from dataclasses import dataclass, asdict, field
from typing import Sequence, Union
import torch

# list of supported objectives which are classification objectives
CLASSIFICATION_LOSSES = ['log', 'xentropy']


@dataclass
class Config:
    # bookkeeping, etc.
    log_interval: int = 10  # number of steps between consecutive log prints
    log_interval_eval: int = None  # same for eval, where None only prints at the end of eval
    save_freq: int = 0   # frequency of saving checkpoints - 0 means no saving
    save_best_checkpoint: bool = False  # save checkpoints with the best performance across fields defined by the filter below
    best_checkpoint_metric: str = 'test/.*/accuracy'  # filter for performance metric to determine best checkpoint and early stopping
    output_dir: str = None  # output directory (submit_multiple.py determines this automagically )
    log_to_stdout: bool = False
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    host_name: str = None  # for bookkeeping - should not be set
    multi_gpu: bool = True
    num_workers: int = 2  # number of cpu workers used for dataloading
    compile_mode: str = None # Can be either 'default', 'reduce-overhead' or 'max-autotune'.

    eval_batch_size: int = None
    max_allowed_loss: float = 1e6  # training stops if loss exceeds this value

    early_stopping_patience: int = None  # number of epochs to stop if eval error does not improve

    # seeds
    seed_model: int = None  # determines model initialization
    seed_training: int = None  # determine training randomness (e.g., batch order)

    # data
    dataset: str = 'libsvm/a9a'  # also supported: 'timm/torch/cifar10', 'timm/tfds/cifar10'
    data_dir: str = 'data'
    data_download: bool = True   # whether to download missing datasets. note: libsvm currently always downloads regardless of the value of this flag
    data_n_examples: dict = None   # for bookkeeping - should not be set
    data_n_classes: int = None   # for bookkeeping - should not be set (automatically inferred when relevant)
    data_splits_map: dict = field(default_factory=lambda: dict(train='train', test='test'))
    # data_splits_map is a dictionary mapping output split names (i.e., how they appear in stats.csv) to split specification in the function used to create datasets.
    # This is particularly useful for tfds, where you can write something like
    # data_splits_map = dict(train='train[:90%]', val='val[90%:]', test='test') in order to create yourself a validation set for parameter selection
    # Note that data_splits_map must contain a key called train (that's your training set) but all other keys are optional and their name can be anything you want
    # It is also possible to set data_splits_map to be string, in which case it is converted to dict via eval

    # model
    model: str = 'linear/no_bias'  # example other options: 'timm/resnet50', 'tiny/wrn_40_2', 'clip/vit_b32', 'linear'
    model_pretrained: bool = False  # only some models support pretrained weights
    model_head_init: str = None  # one of 'zero', 'rand' and None for default value
    model_output_dim: int = None  # will usually be automatically inferred from data_n_classes
    model_input_shape: tuple = None  # can be automatically inferred from model config or data

    # loss
    loss: str = 'log'  # also supported: 'mse'

    # optimization
    batch_size: int = 100
    train_length: int = 100  # how
    train_length_unit: str = 'epoch'  # can also be 'step' or 'sample'
    steps_per_epoch: int = None    # if set, performs eval every fixed number of steps, calling that fixed number an "epoch"
    samples_per_epoch: int = None  # if set, performs eval every fixed number of samples, calling that fixed number an "epoch"
    grad_accum_steps: int = 1   # number of sub-batches for gradient accumulation - the on device batch size will be batch_size // grad_accum_steps
    batch_size_accum: int = None  # on-device batch size; for bookkeeping - should not be set

    optim_alg: str = 'sgd'  # see create_optimizer for supported algorithms
    lr_schedule: str = 'constant' # see create_optimizer for supported scheduler
    base_lr: Union[float, str] = 1.0  # will eval string in order to randomize over base lrs
    weight_decay: float = 0.0
    weight_decay_biases: float = 0.0  # weight decay for any parameter with 'bias' in the name
    momentum: Union[float, str] = 0.9  # for adam-like algoritmhs can specify beta1 and beta2 as strings concatenated by "," for example '0.9,0.999'
    ada_eps: float = 1e-8  # epsilon variable for various adaptime optimization methods

    # averaging
    averaging: str = None  # example options: 'ema_0.999', 'poly_8' (poly_0 is uniform averaging)
import libsvmdata
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize
import os
import sys
import torch

from torch.utils.data import TensorDataset

class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def create_libsvm_dataset(dataset_name, split='train', root=None):
    if root is not None:
        libsvmdata.datasets.DATA_HOME = Path(root) / 'libsvm'

    #todo: support (also?) tfds-like splits, perhaps using sklearn.model_selection.train_test_split
    if split != 'train':
        dataset_name = dataset_name + '_' + split

    with SuppressPrint():
        X, y = libsvmdata.fetch_libsvm(dataset_name)

    # normalize features
    X = normalize(X, axis=1)

    # fix ugly labeling convetions
    unique_y = np.unique(y)
    if tuple(unique_y) == (-1, 1):
        y = (y + 1) // 2
    if unique_y.min() == 1:
        y = y - 1

    # todo: proper y casting for non-classification data
    return TensorDataset(torch.tensor(X.toarray()).float(), torch.LongTensor(y))

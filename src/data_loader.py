import numpy as np
from torchvision.datasets import EMNIST
import torch
from config import DATA_PATH

def load_emnist_digits():
    train_ds=EMNIST(root=DATA_PATH, split='balanced', train=True, download=True)
    test_ds=EMNIST(root=DATA_PATH, split='balanced', train=False, download=True)

    X_train=train_ds.data.numpy()
    y_train=train_ds.targets.numpy()
    X_test=test_ds.data.numpy()
    y_test=test_ds.targets.numpy()

    mask_train=y_train<10
    mask_test=y_test<10

    return X_train[mask_train], y_train[mask_train], X_test[mask_test], y_test[mask_test]

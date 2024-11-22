import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats  # Add this import
from scipy.stats import gaussian_kde
import random


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

CONFIG = {
    'backtest_days': 378,  # ~1.5 years trading days
    'rolling_window': 1008,  # 4-year window
    'confidence_level': 0.95,
    'sequence_length': 252,
    'num_scenarios': 1000,
    'covid_period': {
        'start': '2020-02-18',
        'end': '2020-03-20'
    }
}
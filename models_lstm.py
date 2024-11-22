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


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout)
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        self.vol_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        if self.training:
            noise = torch.randn_like(last_hidden) * 0.01
            last_hidden = last_hidden + noise
        mean = self.mean_head(last_hidden)
        vol = self.vol_head(last_hidden)
        return mean, vol
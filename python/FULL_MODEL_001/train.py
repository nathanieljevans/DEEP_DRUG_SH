import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle
from torch.utils import data
from matplotlib import pyplot as plt
import pandas as pd
import imageio

# ---
import utils
from config import *
import Net
# ---

'''
Global constants are stored in `config.py`
'''

if __name__ == '__main__':

    if REPRODUCIBLE:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

    utils.split_data()

    with open(SPLIT_LABEL_PATH, 'rb') as f:
        label_dict = pickle.load(f)

    utils.print_split_label_dict(label_dict)

    train_labels, test_labels, _, _ = utils.aggregate_labels(label_dict)

    train_gen = data.DataLoader(utils.DrugExpressionDataset(train_labels, return_response_type=True), **train_params)
    test_gen = data.DataLoader(utils.DrugExpressionDataset(test_labels, return_response_type=True), **test_params)

    net = Net.Net(train_gen, test_gen)
    net.train_model()
















##

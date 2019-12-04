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
import random

# ---
import utils
from config import *
import Net
# ---

'''
Global constants are stored in `config.py` in a dict named `params`
'''

if __name__ == '__main__':

    if params['REPRODUCIBLE']:
        torch.manual_seed(params['SEED'])
        np.random.seed(params['SEED'])

    utils.split_data()

    with open(params['SPLIT_LABEL_PATH'], 'rb') as f:
        label_dict = pickle.load(f)

    utils.print_split_label_dict(label_dict)

    train_y_params, train_labels, test_labels, _, _ = utils.aggregate_labels(label_dict)

    #N = 10000
    #train_labels = dict(random.sample(train_labels.items(), N))
    #test_labels = dict(random.sample(test_labels.items(), N))

    train_gen = data.DataLoader(utils.DrugExpressionDataset(train_labels, root_dir=params['DATA_DIR'], return_response_type=True), **params['train_params'])
    test_gen = data.DataLoader(utils.DrugExpressionDataset(test_labels, root_dir=params['DATA_DIR'], return_response_type=True), **params['test_params'])

    net = Net.Net(train_gen, test_gen, params, train_y_params)

    print('pretraining model...')
    net.pretrain_model()

    print('training model...')
    net.train_model()

    print('saving model...')
    net.save_model()
















##

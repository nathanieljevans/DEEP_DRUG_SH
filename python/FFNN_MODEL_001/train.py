###############################################################################
###############################################################################
###############################################################################
'''
THIS IS THE BEATAML ONLY TRAIN FILE
'''
###############################################################################
###############################################################################
###############################################################################

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
import argparse

# ---
import sys
sys.path.append('../UTILS/')
import utils
#from config import *
import Net
# ---

'''
Global constants are stored in `config.py` in a dict named `params`
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get config files path')
    parser.add_argument('--config', type=str, nargs='+',
                        help='config file', dest='config_file')

    args = parser.parse_args() # config_path
    print(f'configuration file (local): {args.config_file[0]}')
    config = __import__(args.config_file[0])
    params = config.params

    if params['REPRODUCIBLE']:
        torch.manual_seed(params['SEED'])
        np.random.seed(params['SEED'])

    utils.split_data(label_path=params['LABEL_PATH'], \
                    split_label_path=params['SPLIT_LABEL_PATH'], \
                    splits=(params['TRAIN_PROP'],params['TEST_PROP'],params['VAL_PROP']), \
                    resplit_data=params['RESPLIT_DATA'], \
                    aml_pats=params['N_BEATAML_PATIENTS_EXCLUSIVE_TO_TEST'])

    with open(params['SPLIT_LABEL_PATH'], 'rb') as f:
        label_dict = pickle.load(f)

    utils.print_split_label_dict(label_dict)

    train_y_params, train_labels, test_labels, _, _ = utils.aggregate_labels(label_dict, keep=params['RESP_TYPES'].keys())

    #N = 10000
    #train_labels = dict(random.sample(train_labels.items(), N))
    #test_labels = dict(random.sample(test_labels.items(), N))

    train_gen = data.DataLoader(utils.DrugExpressionDataset(train_labels, \
                                                root_dir=params['DATA_DIR'], \
                                                resp_types=params['RESP_TYPES'], \
                                                return_response_type=True), \
                                                **params['train_params'])

    test_gen = data.DataLoader(utils.DrugExpressionDataset(test_labels, \
                                                root_dir=params['DATA_DIR'], \
                                                resp_types=params['RESP_TYPES'], \
                                                return_response_type=True), \
                                                **params['test_params'])

    net = Net.Net(train_gen, test_gen, params, train_y_params)

    print('pretraining model...')
    net.pretrain_model()

    print('training model...')
    net.train_model()

    print('saving model...')
    net.save_model()
















##

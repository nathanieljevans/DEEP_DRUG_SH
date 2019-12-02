'''
We want to answer the question:
    For the same drug target in each dataset, what is the correllation of the response variables.

'''

import sys
sys.path.insert(1, './FULL_MODEL_001/')

import pickle
from matplotlib import pyplot as plt
import numpy as np
from config import *    # params stored here
import utils
import pandas as pd
from torch.utils import  data

if __name__ == '__main__':

    with open('./../data_pytorch/split_label_dict.pkl', 'rb') as f:
        label_dict = pickle.load(f)

    targets = {'dataset' : [], 'target' : [], 'response' : []}
    for i,dataset in enumerate([dataset for dataset in params['RESP_TYPES']]):
        test = label_dict[dataset]['test']
        gen = data.DataLoader(utils.DrugExpressionDataset(test, root_dir='./../data_pytorch/tensors/', return_response_type=True), **{'batch_size':1, 'shuffle':False,'num_workers':0})
        ii = 0
        for X,y,resp_type,resp_selector in gen:
            ii+=X.size(0)
            #if ii > 1000:
            #    break
            if ii/X.size(0) % 1000 == 0:
                print(f'predicting {dataset} ...[{ii}/{len(gen.dataset)}]', end='\r')
            targets['dataset'].append(dataset)
            targets['target'].append(hash(str(X.numpy()[:,1])))
            targets['response'].append(y.numpy()[0])
        print()

    df = pd.DataFrame(targets)
    df = df.groupby(['dataset', 'target']).agg({'response':'mean'})
    df = df.reset_index()
    print(df.head())

    corr = {'dataset1':[],'dataset2':[],'corr':[]}
    for i, dataset1 in enumerate([dataset for dataset in params['RESP_TYPES']]):
        for j,dataset2 in enumerate([dataset for dataset in params['RESP_TYPES']][(i+1):]):
            tmp = df[df['dataset'].isin([dataset1, dataset2])]
            intersection_targets = np.intersect1d(tmp[tmp['dataset'] == dataset1].target.values, tmp[tmp['dataset'] == dataset2].target.values)
            tmp = tmp[tmp['target'].isin(intersection_targets)].pivot(index='target', columns='dataset', values='response')
            print(f'[{dataset1}, {dataset2}] intersecting targets: {tmp.values.shape[0]}')
            if tmp.values.shape[0] > 0
                corr['dataset1'].append(dataset1)
                corr['dataset2'].append(dataset2)
                corr['corr'].append(np.correlate(tmp.values[:,0], tmp.values[:,1]))

    print(corr)






#

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

    targets = {'dataset' : [], 'target' : [], 'response' : [], 'expr' : []}
    for i,dataset in enumerate([dataset for dataset in params['RESP_TYPES']]):
        test = label_dict[dataset]['test']
        gen = data.DataLoader(utils.DrugExpressionDataset(test, root_dir='./../data_pytorch/tensors/', return_response_type=True), **{'batch_size':1, 'shuffle':False,'num_workers':0})
        ii = 0
        for X,y,resp_type,resp_selector in gen:
            ii+=X.size(0)
            if ii > 1000:
                break
            if ii/X.size(0) % 1000 == 0:
                print(f'predicting {dataset} ...[{ii}/{len(gen.dataset)}]', end='\r')
            targets['dataset'].append(dataset)
            targets['target'].append(hash(str(X.numpy()[:,1])))
            targets['response'].append(y.numpy()[0])
            targets['expr'].append(hash(str(np.round(X.numpy()[:,0]))))

        print()

    df = pd.DataFrame(targets)
    df = df.groupby(['dataset', 'target', 'expr']).agg({'response':'mean'})
    df = df.reset_index()

    print(df.head())

    f, axes = plt.subplots(5,3,figsize=(15,10))
    corr = {'dataset1':[],'dataset2':[],'corr':[]}
    ii = 0
    for i, dataset1 in enumerate([dataset for dataset in params['RESP_TYPES']]):
        for j,dataset2 in enumerate([dataset for dataset in params['RESP_TYPES']][(i+1):]):
            print(f'{dataset1} | {dataset2}')
            tmp = df[df['dataset'].isin([dataset1, dataset2])]
            intersection_targets = np.intersect1d(tmp[tmp['dataset'] == dataset1].target.values, tmp[tmp['dataset'] == dataset2].target.values)
            intersection_expr = np.intersect1d(tmp[tmp['dataset'] == dataset1].expr.values, tmp[tmp['dataset'] == dataset2].expr.values)
            print(intersection_expr)
            tmp = tmp[tmp['target'].isin(intersection_targets)]
            print(tmp.head())
            tmp = tmp[tmp['expr'].isin(intersection_expr)]
            print(tmp.head())
            tmp = tmp.pivot(index=['target','expr'], columns='dataset', values='response')
            print(f'[{dataset1}, {dataset2}] intersecting targets: {tmp.values.shape[0]}')
            if tmp.values.shape[0] > 10:
                corr['dataset1'].append(dataset1)
                corr['dataset2'].append(dataset2)
                cor = np.correlate(tmp.values[:,0], tmp.values[:,1])
                corr['corr'].append(cor)

                axes.flat[ii].plot(tmp.values[:,0], tmp.values[:,1], 'r.', alpha=0.1)

                axes.flat[ii].set_title(f'{dataset1}-{dataset2} [{cor[0]:.2f}]', fontsize=10)
                axes.flat[ii].set_xlabel('response1', fontsize=10)
                axes.flat[ii].set_ylabel(f'response2', fontsize=10)

                ii+=1

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.75)
    plt.savefig('./dataset_response_comparison_plot.png')
    print(corr)






#

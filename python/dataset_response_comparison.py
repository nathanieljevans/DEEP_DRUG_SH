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
import statsmodels.api as sm
from scipy.stats import zscore

def mean_na_remove(x):
    x = x[~np.isnan(x)]
    return np.mean(x)

def lm(x1, x2):
    try:
        x = np.linspace(min(x1),max(x1),10)
        x1 = sm.add_constant(x1)
        model = sm.OLS(x2, x1)
        results = model.fit()

        y = results.predict(sm.add_constant(x))

        p = results.f_test(np.identity(2)).pvalue

        return p, x, y, results.params[1]
    except:
        raise
        return -1, [0], [0]

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
            #if ii > 1000:
            #    break
            if ii/X.size(0) % 1000 == 0:
                print(f'predicting {dataset} ...[{ii}/{len(gen.dataset)}]', end='\r')
            targets['dataset'].append(dataset)
            targets['target'].append(hash(str(X.numpy()[:,1])))
            targets['response'].append(y.numpy()[0])
            targets['expr'].append(hash(str(np.round(X.numpy()[:,0], decimals=0))))
        print()

    df = pd.DataFrame(targets)
    df = df.groupby(['dataset', 'target', 'expr']).agg({'response':mean_na_remove})
    df = df.reset_index()

    print(df.head())

    f, axes = plt.subplots(5,3,figsize=(15,12))
    corr = {'dataset1':[],'dataset2':[],'corr':[]}
    ii = 0
    for i, dataset1 in enumerate([dataset for dataset in params['RESP_TYPES']]):
        for j,dataset2 in enumerate([dataset for dataset in params['RESP_TYPES']][(i+1):]):
            try:
                tmp = df[df['dataset'].isin([dataset1, dataset2])]
                intersection_targets = np.intersect1d(tmp[tmp['dataset'] == dataset1].target.values, tmp[tmp['dataset'] == dataset2].target.values)
                intersection_expr = np.intersect1d(tmp[tmp['dataset'] == dataset1].expr.values, tmp[tmp['dataset'] == dataset2].expr.values)
                tmp = tmp[tmp['target'].isin(intersection_targets)]
                tmp = tmp[tmp['expr'].isin(intersection_expr)]
                tmp = tmp.pivot_table(index=['target','expr'], columns='dataset', values='response', aggfunc=mean_na_remove)
                tmp = tmp.dropna()
                print(f'[{dataset1}, {dataset2}] intersecting targets: {tmp.values.shape[0]}')
                if tmp.values.shape[0] > 10:
                    X1 = zscore(tmp.values[:,0])
                    X2 = zscore(tmp.values[:,1])
                    p, xx, yy, B = lm(X1, X2)
                    corr['dataset1'].append(dataset1)
                    corr['dataset2'].append(dataset2)
                    cor = np.correlate(X1, X2)
                    corr['corr'].append(cor)

                    axes.flat[ii].plot(X1, X2, 'r.', alpha=0.2)
                    axes.flat[ii].plot(xx, yy, 'b--', label=f'pval: {p:.2g} \nslope: {B:.2f}')
                    axes.flat[ii].legend()
                    axes.flat[ii].set_title(f'{dataset1}-{dataset2} [cor: {cor[0]:.2f}]', fontsize=10)
                    axes.flat[ii].set_xlabel('response1', fontsize=10)
                    axes.flat[ii].set_ylabel(f'response2', fontsize=10)

                    ii+=1
            except:
                print(f'Failed: {dataset1} - {dataset2}')
                raise

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.75)
    plt.savefig('../Dataset_Comparison/dataset_response_comparison_plot.png')
    print(corr)






#

'''
THIS IS THE BEATAML ONLY TEST FILE
'''

import sys
sys.path.append(r'C:\Users\natha\Documents\DEEP_DRUG_SH\python\UTILS')
import pickle
from matplotlib import pyplot as plt
import numpy as np
from config import *    # params stored here
import utils
import pandas as pd
from torch.utils import  data

if __name__ == '__main__':

    with open(f"{params['MODEL_OUT_DIR']}/{params['NAME']}/model.pkl", 'rb') as f:
        net = pickle.load(f)

    with open(params['SPLIT_LABEL_PATH'], 'rb') as f:
        label_dict = pickle.load(f)

    plt.gcf()
    f, axes = plt.subplots(4,2,figsize=(15,12))

    for i,dataset in enumerate([dataset for dataset in params['RESP_TYPES']]):
        test = label_dict[dataset]['test']
        gen = data.DataLoader(utils.DrugExpressionDataset(test, root_dir=params['DATA_DIR'], return_response_type=True), **{'batch_size':10000, 'shuffle':False,'num_workers':0})
        yhats = []
        ys = []
        ii = 0
        for X,y,resp_type,resp_selector in gen:
            ii+=X.size(0)
            print(f'predicting {dataset} ...[{ii}/{len(gen.dataset)}]', end='\r')
            yhats += net.predict(X, resp_type, resp_selector).tolist()
            ys += y.tolist()
            #if ii > 1000:
            #    break
        print()

        mse = np.mean((np.array(ys) - np.array(yhats))**2)
        df = pd.DataFrame({'y':ys, 'yhat':yhats})
        df.sort_values(by='y', inplace=True)

        axes.flat[-1].text(0.1,i*0.125,f'{dataset} mse: {" "*(30-len(dataset))} {mse:.3f}',  fontdict={'size': 10, 'color':  'red'})
        axes.flat[-1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axes.flat[-1].tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
        alpha_ = 284./len(ys)

        #axes.flat[i].plot(np.log10(((df['y']-df['yhat'])**2).values), 'g--', label='log10 Quadratic Error', alpha=0.2)
        axes.flat[i].plot(df['yhat'].values, 'b.', label='predicted', alpha=alpha_)
        axes.flat[i].plot(df['y'].values, 'r.', label='true', alpha=alpha_)

        axes.flat[i].set_title(f'Predictions: {dataset}', fontsize=10)
        axes.flat[i].set_xlabel('Sorted observations', fontsize=10)
        axes.flat[i].set_ylabel(f'Response', fontsize=10)
        axes.flat[i].legend()

    for pos in ['right','top','bottom','left']:
        axes.flat[-1].spines[pos].set_visible(False)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=0.45)
    plt.savefig(f"{params['MODEL_OUT_DIR']}/{params['NAME']}/test_predictions_plot.png")

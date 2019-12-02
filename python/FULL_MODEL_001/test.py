'''

'''
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
    f, axes = plt.subplots(3,3,figsize=(12,15))

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
        print()

        df = pd.DataFrame({'y':ys, 'yhat':yhats})
        df.sort_values(by='y', inplace=True)

        #axes.flat[i].plot(np.log10(((df['y']-df['yhat'])**2).values), 'g--', label='log10 Quadratic Error', alpha=0.2)
        axes.flat[i].plot(df['y'].values, 'r.', label='true', alpha=0.5)
        axes.flat[i].plot(df['yhat'].values, 'b.', label='predicted', alpha=0.05)

        axes.flat[i].set_title(f'Predictions: {dataset}', fontsize=10)
        axes.flat[i].set_xlabel('Sorted observations', fontsize=10)
        axes.flat[i].set_ylabel(f'Response', fontsize=10)
        axes.flat[i].legend()

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.35)
    plt.savefig(f"{params['MODEL_OUT_DIR']}/{params['NAME']}/test_predictions_plot.png")

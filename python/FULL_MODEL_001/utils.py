from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from config import *
import os
import pickle
import numpy as np
import torch

# plot and show learning process
#from : https://medium.com/@benjamin.phillips22/simple-regression-with-neural-networks-in-pytorch-313f06910379
class Training_Progress_Plotter:
    def __init__(self, figsize = (12,10)):
        '''
        '''
        self.axes, self.fig = plt.subplots(1,2, figsize = figsize)
        self.images = []

    def update(tr_ys, tr_yhats, tst_ys, tst_yhats, epoch, tr_loss, tst_loss):
        '''
        Record the training progress at each epoch.
        '''
        self.axes[0].cla()

        ######### TRAIN #########
        tr_df = pd.DataFrame({'y':tr_ys, 'yhat':tr_yhats})
        tr_df.sort_values(by='y', inplace=True)

        self.axes[0].plot(tr_df.values[:,0], 'ro', label='true', alpha=0.5)
        self.axes[0].plot(tr_df.values[:,1], 'bo', label='predicted', alpha=0.5)

        self.axes[0].set_title('Regression Analysis [Training Set]', fontsize=15)
        self.axes[0].set_xlabel('Sorted observations', fontsize=24)
        self.axes[0].set_ylabel('AUC', fontsize=24)

        self.axes[0].text(100, 30, 'Epoch = %d' % epoch, fontdict={'size': 24, 'color':  'red'})
        self.axes[0].text(100, 50, 'Loss = %.4f' % loss, fontdict={'size': 24, 'color':  'red'})

        ######### TEST #########

        self.axes[1].cla()
        tst_df = pd.DataFrame({'y':tst_ys, 'yhat':tst_yhats})
        tst_df.sort_values(by='y', inplace=True)

        self.axes[1].plot(tst_df.values[:,0], 'ro', label='true', alpha=0.5)
        self.axes[1].plot(tst_df.values[:,1], 'bo', label='predicted', alpha=0.5)
        plt.legend()

        self.axes[1].set_title('Regression Analysis [Validation Set]', fontsize=15)
        self.axes[1].set_xlabel('Sorted observations', fontsize=24)
        self.axes[1].set_ylabel('AUC', fontsize=24)

        self.axes[1].text(100, 30, 'Epoch = %d' % epoch, fontdict={'size': 24, 'color':  'red'})
        self.axes[1].text(100, 50, 'Loss = %.4f' % loss, fontdict={'size': 24, 'color':  'red'})

        # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        self.images.append(image)

    def save_gif(self, name, path):
        '''
        '''
        imageio.mimsave(f'{name}_training.gif', self.images, fps=10)

class saver_and_early_stopping:
    '''
    '''
    def __init__(self, max_epoch, lr, do, arch, wd, name='FCNN-AML', early_stop_patience = 10, save_path='../data_pytorch/'):
        ''''''
        self.lr = lr
        self.do = do
        self.arch = arch
        self.wd = wd
        self.name = name
        self.loss = np.inf
        self.ii = 0
        self.patience = early_stop_patience
        self.max_epoch = max_epoch

        if not os.path.exists(save_path + 'model_' + name):
            os.mkdir(save_path + 'model_' + name)

        self.fpath = save_path + 'model_' + name

    def update(self, loss, epoch, model):
        '''
        '''
        if epoch == (self.max_epoch - 1):
            torch.save(model.state_dict(), self.fpath + f'/loss={self.loss:.2f}_epoch={epoch}_lr={self.lr}_do={self.do}_arch={self.arch}_wd={self.wd}.pt')
            self.fname = self.fpath + f'/loss={self.loss}_epoch={epoch}_lr={self.lr}_do={self.do}_arch={self.arch}_wd={self.wd}'
            return True
        if self.ii > self.patience:
            torch.save(model.state_dict(), self.fpath + f'/loss={self.loss:.2f}_epoch={epoch}_lr={self.lr}_do={self.do}_arch={self.arch}_wd={self.wd}.pt')
            self.fname = self.fpath + f'/loss={self.loss}_epoch={epoch}_lr={self.lr}_do={self.do}_arch={self.arch}_wd={self.wd}'
            return False
        if loss < self.loss:
            self.loss = loss
            self.ii = 0
        else:
            self.ii += 1
        return True

    def get_fname(self):
        ''''''
        return self.fname


class DrugExpressionDataset(Dataset):
    '''
    '''
    def __init__ (self, label_dict, root_dir=DATA_DIR, return_response_type=False):
        '''
        '''
        self.index = list(label_dict.keys())
        self.labels = label_dict
        self.root = root_dir
        self.ret_resp_type = return_response_type

    def __len__(self):
        '''
        '''
        return len(self.labels.keys())

    def __getitem__(self, index, response_type=False):
        '''
        '''
        fid = self.index[index]
        X = torch.load(f'{self.root}/{fid}.pt')
        _id, id_type, resp_type, response = self.labels[fid]

        if self.ret_resp_type:
            return X, response, resp_type
        else:
            return X, response


def print_split_label_dict(label_dict):
    '''
    '''
    for resp_type in label_dict:
        for sset in label_dict[resp_type]:
            pp = resp_type + ' '*(20 - len(resp_type))
            print(f'set sizes: {pp} \t-> {sset}  \t-> {len(label_dict[resp_type][sset])}')

def split_data():
    assert os.path.exists(LABEL_PATH), 'label dictionary path does not exist, have you run `pytorch_data_separation.py` locally?'
    assert sum((TRAIN_PROP, TEST_PROP, VAL_PROP)) == 1, 'split proportions do not sum to 1; update `config.py`'
    if (os.path.exists(SPLIT_LABEL_PATH) and not RESPLIT_DATA):
        print('Train, Test, Validation sets have already been split, exiting...')
        return None

    with open(LABEL_PATH, 'rb') as f: label_dict = pickle.load(f)

    label_dict2 = dict()
    AML_HOLDOUTS = set()
    ii = 0
    for fid in label_dict:
        _id, id_type, resp_type, response = label_dict[fid]
        fid = str(fid)

        if resp_type not in label_dict2:
            label_dict2[resp_type] = dict()

        p = np.random.rand()
        if (resp_type == 'beatAML_AUC' and ii < N_BEATAML_PATIENTS_EXCLUSIVE_TO_TEST) or (_id in AML_HOLDOUTS):
            if ii == 0:
                label_dict2[resp_type]['PAT_HOLDOUT'] = {fid : (_id, id_type, resp_type, response)}
            else:
                label_dict2[resp_type]['PAT_HOLDOUT'][fid] = (_id, id_type, resp_type, response)

            AML_HOLDOUTS.add(_id)
            ii += 1

        if (p < TRAIN_PROP): # add to training set
            if 'train' in label_dict2[resp_type]:
                label_dict2[resp_type]['train'][fid] = (_id, id_type, resp_type, response)
            else:
                label_dict2[resp_type]['train'] = {fid:(_id, id_type, resp_type, response)}

        elif (p < TRAIN_PROP + TEST_PROP): # add to test set
            if 'test' in label_dict2[resp_type]:
                label_dict2[resp_type]['test'][fid] = (_id, id_type, resp_type, response)
            else:
                label_dict2[resp_type]['test'] = {fid:(_id, id_type, resp_type, response)}
        else: # add to validation set
            if 'val' in label_dict2[resp_type]:
                label_dict2[resp_type]['val'][fid] = (_id, id_type, resp_type, response)
            else:
                label_dict2[resp_type]['val'] = {fid:(_id, id_type, resp_type, response)}

    print_split_label_dict(label_dict2)

    with open(SPLIT_LABEL_PATH, 'wb') as f:
        pickle.dump(label_dict2, f)


def aggregate_labels(label_dict):

    train_labels = dict()
    test_labels = dict()
    val_labels = dict()

    for dataset in label_dict:
        for split in label_dict[dataset]:
            if split == 'train':
                train_labels = {**train_labels, **label_dict[dataset][split]}
            if split == 'test':
                test_labels = {**test_labels, **label_dict[dataset][split]}
            if split == 'val':
                val_labels = {**val_labels, **label_dict[dataset][split]}

    aml_holdout = label_dict['beatAML_AUC']['PAT_HOLDOUT']

    print('aggregated labels:')
    print(f'\ttrain set: \t {len(train_labels)}')
    print(f'\ttest set: \t {len(test_labels)}')
    print(f'\tval set: \t {len(val_labels)}')
    print(f'\taml holdout set:\t {len(aml_holdout)}')

    return train_labels, test_labels, val_labels, aml_holdout

###############################################################################
###############################################################################
###############################################################################
'''
THIS IS THE SHARED NET FILE
'''
###############################################################################
###############################################################################
###############################################################################

import torch
import torch.nn.functional as F
import numpy as np
import time
import sys
sys.path.append(r'C:\Users\natha\Documents\DEEP_DRUG_SH\python\UTILS')
import utils
from matplotlib import pyplot as plt
import os
import pickle
import copy

class Net(torch.nn.Module):

    def __init__(self, train_gen, test_gen, model_params, Y_params):
        super(Net, self).__init__()

        self.plotter = utils.Training_Progress_Plotter()
        self.params = model_params
        self.train_gen = train_gen
        self.test_gen = test_gen
        self.Y_params = Y_params
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(1,
                                            self.params['NCONVS'],
                                            kernel_size=(1,2),
                                            stride=1,
                                            padding=0),
                                    torch.nn.ReLU())

        self.do = torch.nn.Dropout(self.params['DO'])
        self.fc1 = torch.nn.Linear(self.params['NGENES']*self.params['NCONVS'], self.params['H1'])
        self.bn1 = torch.nn.BatchNorm1d(self.params['H1'])
        self.fc2 = torch.nn.Linear(self.params['H1'], self.params['H2'])
        self.bn2 = torch.nn.BatchNorm1d(self.params['H2'])
        self.fc3 = torch.nn.Linear(self.params['H2'], self.params['H3'])
        self.bn3 = torch.nn.BatchNorm1d(self.params['H3'])
        self.DSP = [torch.nn.Linear(self.params['H3'], self.params['DH']) for _ in self.params['RESP_TYPES']]
        self.out = [torch.nn.Linear(self.params['DH'], 1) for _ in self.params['RESP_TYPES']]

        self.AE_do = torch.nn.Dropout(self.params['PRETRAIN_DO'])
        self.AE_fc4 = torch.nn.Linear(self.params['H3'], self.params['H2'])
        self.AE_fc5 = torch.nn.Linear(self.params['H2'], self.params['H1'])
        self.AE_fc6 = torch.nn.Linear(self.params['H1'], 2*self.params['NGENES'])

        if not os.path.exists(f"{self.params['MODEL_OUT_DIR']}/{self.params['NAME']}"):
            os.mkdir(f"{self.params['MODEL_OUT_DIR']}/{self.params['NAME']}")

    def forward(self, x):
        x = x.unsqueeze(1)

        a0 = F.relu(self.layer1(x))
        a0 = a0.reshape(a0.size(0), -1)    # Conv2D

        a1 = self.do(self.bn1(F.relu(self.fc1(a0))))                    # FC layer 1

        a2 = self.do(self.bn2(F.relu(self.fc2(a1))))                     # FC layer 2

        a3 = self.do(self.bn3(F.relu(self.fc3(a2))))                     # FC layer 3

        aDS = [F.leaky_relu(FC(a3)) for FC in self.DSP]
        o = [out(a4) for a4, out in zip(aDS, self.out)]
        endout = torch.zeros(x.size(0), len(self.params['RESP_TYPES'].keys()),dtype=torch.float)
        for i,oo in enumerate(o): endout[:,i] = oo.squeeze()

        return endout

    def pretrain_forward(self, x):
        x = x.unsqueeze(1)
        #x = self.do(x)
        ################################ ENCODE ################################
        a0 = F.relu(self.layer1(x))
        a0 = a0.reshape(a0.size(0), -1)                         # Conv2D
        a1 = self.AE_do(self.bn1(F.relu(self.fc1(a0))))                     # FC layer 1
        a2 = self.AE_do(self.bn2(F.relu(self.fc2(a1))))                     # FC layer 2
        a3 = self.AE_do(self.bn3(F.relu(self.fc3(a2))))                     # FC layer 3
        ################################ DECODE ################################
        a4 = self.AE_do(F.relu(self.AE_fc4(a3)))
        a5 = self.AE_do(F.relu(self.AE_fc5(a4)))
        a6 = self.AE_do(F.relu(self.AE_fc6(a5)))
        a6 = a6.reshape(a6.size(0), self.params['NGENES'], 2)
        return a6

    def pretrain_model(self):
        '''
        '''
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        optim = torch.optim.Adam(self.parameters(recurse=True), lr=self.params['PRETRAIN_LR'], weight_decay=self.params['PRETRAIN_WD'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min',
                                                            factor=0.1, patience=25,
                                                            verbose=True, threshold=10,
                                                            threshold_mode='rel', cooldown=0,
                                                            min_lr=1e-5, eps=1e-08)
        mse = []
        for epoch in range(self.params['PRETRAIN_EPOCHS']):
            total_loss = 0
            self.train()
            ii = 0
            for X,y,_,resp_selector in self.train_gen:
                ii += X.size(0)
                X = X.to(device, dtype=torch.float)
                xhat = self.pretrain_forward(X)
                loss = utils.AE_LOSS(xhat, X, gamma=self.params['PRETRAIN_MSE_WEIGHT'])
                optim.zero_grad()
                loss.backward()
                optim.step()
                total_loss += loss.detach().numpy()
                if self.params['PRINT_MID_EPOCH_INFO']: print(f'EPOCH: {epoch} [training set] \t mse: [{(total_loss/ii):.4f}]  \t |  Epoch progress: [{ii}/{len(self.train_gen.dataset)}] {"."*int(ii/(len(self.train_gen.dataset)/10))}', end = '\t\t\t\n')
                scheduler.step(total_loss)
                mse.append((total_loss/ii))
                self.plot_mse(mse, 'pretraining_mse')

    def plot_mse(self, mse, name):
        plt.figure()
        plt.plot((self.params['train_params']['batch_size']/len(self.test_gen.dataset))*np.arange(len(mse)), mse, 'r-')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title(name)
        plt.savefig(f"{self.params['MODEL_OUT_DIR']}/{self.params['NAME']}/{name}.png")
        plt.close('all')

    def train_model(self):

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        optim = torch.optim.Adam(self.parameters(recurse=True), lr=self.params['LEARNING_WEIGHT'], weight_decay=self.params['WEIGHT_DECAY'])
        loss_func = torch.nn.MSELoss(reduction='sum')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min',
                                                            factor=0.1, patience=self.params['LR_DECAY_PATIENCE'],
                                                            verbose=False, threshold=100,
                                                            threshold_mode='rel', cooldown=0,
                                                            min_lr=1e-5, eps=1e-08)

        self.recorder={'train':{'total_loss':[], 'mse':[], 'batch-mse':[]}, 'test':{'mse':[], 'batch-mse':[]}}
        MAX_EPOCH = self.params['EPOCHS']
        tr_mse = []
        for epoch in range(MAX_EPOCH):

            total_loss = 0
            yhats=[]
            ys=[]
            self.train()
            ii = 0
            for X,y,_,resp_selector in self.train_gen:
                ii+=X.size(0)
                X,y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.float)
                yhat = self.forward(X)
                y2 = yhat.detach().numpy().copy()           # only use gradient by one route
                y2[resp_selector==1] = y
                y2 = torch.FloatTensor(y2)
                loss = loss_func(yhat, y2)
                optim.zero_grad()
                loss.backward()
                optim.step()
                scheduler.step(total_loss)
                total_loss += loss.detach().numpy()
                yhats += yhat.data.numpy()[resp_selector==1].ravel().tolist()
                ys += y.data.numpy().ravel().tolist()
                self.recorder['train']['batch-mse'].append(loss.detach().numpy()/X.size(0))
                tr_mse.append(total_loss/ii)
                self.plot_mse(tr_mse, 'training_sse')
                if self.params['PRINT_MID_EPOCH_INFO']: print(f'Train set... mse: [{loss.detach().numpy()/(X.size(0)):.4f}] | Epoch progress: [{ii}/{len(self.train_gen.dataset)}] {"."*int(ii/(len(self.train_gen.dataset)/10))}', end = '\t\t\t\n')

            self.recorder['train']['total_loss'].append(total_loss)
            self.recorder['train']['mse'].append(total_loss/len(self.train_gen.dataset))
            self.eval()
            test_total_loss = 0
            test_yhats = []
            test_ys = []
            jj = 0
            for X,y,_,resp_selector in self.test_gen:
                jj += X.size(0)
                X,y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.float)
                yhat = self.forward(X)
                y2 = yhat.detach().numpy().copy()           # only use gradient by one route
                y2[resp_selector==1] = y
                y2 = torch.FloatTensor(y2)
                loss = loss_func(yhat, y2).detach().numpy()
                test_total_loss += loss
                test_yhats += yhat.data.numpy()[resp_selector==1].ravel().tolist()
                test_ys += y.data.numpy().ravel().tolist()
                self.recorder['test']['batch-mse'].append(loss/X.size(0))
                if self.params['PRINT_MID_EPOCH_INFO']: print(f'Test set... mse: [{loss/(X.size(0)):.4f}] | Epoch progress: [{jj}/{len(self.test_gen.dataset)}] {"."*int(jj/(len(self.test_gen.dataset)/10))}', end = '\t\t\t\n')

            self.recorder['test']['mse'].append(test_total_loss/len(self.test_gen.dataset))

            self.plotter.update(tr_ys=ys, tr_yhats=yhats,
                                tst_ys=test_ys, tst_yhats=test_yhats,
                                epoch=epoch, tr_loss=self.recorder['train']['mse'][-1],
                                tst_loss=self.recorder['test']['mse'][-1])

            if ((epoch + 1) % self.params['PRINT_EVERY']) == 0:
                print()
                print(f'>>> END EPOCH {epoch+1}/{MAX_EPOCH} || train total loss: {total_loss:.2f} | train mse: {total_loss/len(self.train_gen.dataset):.6f} || test total_loss: {test_total_loss:.4g} | test mse: {test_total_loss / len(self.test_gen.dataset):.4g} \t\t\t')

            if (self.params['SAVE_MODEL_EVERY'] != -1) and ((epoch + 1) % self.params['SAVE_MODEL_EVERY'] == 0):
                print('saving model.')
                self.save_model()
                self.plotter.save_gif(f"{self.params['MODEL_OUT_DIR']}/{self.params['NAME']}/training_fit.gif")
        print()
        print('training complete...')
        self.plotter.save_gif(f"{self.params['MODEL_OUT_DIR']}/{self.params['NAME']}/training_fit.gif")
        self.plot_training_loss()

    def plot_training_loss(self):

        plt.figure(figsize=(12,8))
        plt.plot(np.linspace(0, len(self.recorder['train']['mse'])-1, len(self.recorder['train']['batch-mse'])),
                    self.recorder['train']['batch-mse'],
                    'b--', label='train batch-mse', alpha=0.2)
        plt.plot(self.recorder['test']['mse'], 'r-', label='test mse')
        #plt.plot(np.linspace(0, len(self.recorder['test']['mse'])-1, len(self.recorder['test']['batch-mse'])),
        #            self.recorder['test']['batch-mse'],
        #            'r--', label='test batch-mse', alpha=0.75)
        plt.plot(self.recorder['train']['mse'], 'b-', label='train mse')

        plt.xlabel('epochs')
        plt.ylabel('Sum of Square Errors')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig(f"{self.params['MODEL_OUT_DIR']}/{self.params['NAME']}/training_loss_plot.png")

    def unscale_y(self, ys, resp_type):
        '''

        '''
        std_ = np.array([self.Y_params[rt]['std'] for rt in resp_type])
        mean_ = np.array([self.Y_params[rt]['mean'] for rt in resp_type])
        return (ys * std_) + mean_

    def predict(self, X, resp_type, resp_selector):
        '''
        y <np.array>
        resp_type <np.array}
        '''
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        self.eval()
        X = X.to(device, dtype=torch.float)
        yhat = self.forward(X).detach().numpy()
        #resp_selector = np.array([np.array([True if i == self.params['RESP_TYPES'][x] else False for i,x in enumerate(resp_type)])])
        yhat = yhat[resp_selector==1]
        yhat = self.unscale_y(yhat, resp_type)
        return yhat

    def save_model(self):
        '''

        '''
        with open(f"{self.params['MODEL_OUT_DIR']}/{self.params['NAME']}/model.pkl", 'wb') as f:
            pickle.dump(copy.copy(self), f)

    def save_params(self):
        '''
        '''
        with open(f"{self.params['MODEL_OUT_DIR']}/{self.params['NAME']}/config_params.txt", 'w') as f:
            for param in self.params:
                f.write(f'{param}:  {self.params[param]}\n')













#

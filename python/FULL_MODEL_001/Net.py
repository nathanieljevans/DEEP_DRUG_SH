from config import *
import torch
import torch.nn.functional as F
import numpy as np
import time
import utils
from matplotlib import pyplot as plt

class Net(torch.nn.Module):

    def __init__(self, train_gen, test_gen):
        super(Net, self).__init__()

        self.plotter = utils.Training_Progress_Plotter()

        self.train_gen = train_gen
        self.test_gen = test_gen

        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(1,
                                            NCONVS,
                                            kernel_size=(1,2),
                                            stride=1,
                                            padding=0),
                                    torch.nn.ReLU())
        self.do = torch.nn.Dropout(DO)
        self.fc1 = torch.nn.Linear(NGENES*NCONVS, H1)
        self.bn1 = torch.nn.BatchNorm1d(H1)
        self.fc2 = torch.nn.Linear(H1, H2)
        self.bn2 = torch.nn.BatchNorm1d(H2)
        self.fc3 = torch.nn.Linear(H2, H3)
        self.bn3 = torch.nn.BatchNorm1d(H3)
        self.DSP = [torch.nn.Linear(H3, DH) for _ in range(N_DATATYPES)]
        self.out = [torch.nn.Linear(DH, 1) for _ in range(N_DATATYPES)]

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.do(x)

        a0 = F.relu(self.layer1(x))
        a0 = a0.reshape(a0.size(0), -1)    # Conv2D

        a1 = self.bn1(F.relu(self.fc1(a0)))                    # FC layer 1

        a2 = self.bn2(F.relu(self.fc2(a1)))                     # FC layer 2

        a3 = self.bn3(F.relu(self.fc3(a2)))                     # FC layer 3

        aDS = [F.relu(FC(a3)) for FC in self.DSP]
        o = [out(a4) for a4, out in zip(aDS, self.out)]
        endout = torch.zeros(x.size(0), N_DATATYPES,dtype=torch.float)
        for i,oo in enumerate(o): endout[:,i] = oo.squeeze()

        return endout


    def train_model(self):

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        optim = torch.optim.Adam(self.parameters(recurse=True), lr=LEARNING_WEIGHT, weight_decay=WEIGHT_DECAY)
        loss_func = torch.nn.MSELoss(reduction='sum')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min',
                                                            factor=0.1, patience=2,
                                                            verbose=True, threshold=1000,
                                                            threshold_mode='rel', cooldown=0,
                                                            min_lr=1e-5, eps=1e-08)

        self.recorder={'train':{'total_loss':[], 'mse':[], 'batch-mse':[]}, 'test':{'mse':[], 'batch-mse':[]}}

        for epoch in range(EPOCHS):
            total_loss = 0
            mse = 0
            yhats=[]
            ys=[]
            self.train()
            ii = 0
            for X, y, resp_selector in self.train_gen:
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
                total_loss += loss.detach().numpy()
                yhats += yhat.data.numpy()[resp_selector==1].ravel().tolist()
                ys += y.data.numpy().ravel().tolist()
                self.recorder['train']['batch-mse'].append(loss.detach().numpy()/X.size(0))
                print(f'Train set... \t mse: [{loss.detach().numpy()/(X.size(0)):.4f}]  \t |  Epoch progress: [{ii}/{len(self.train_gen.dataset)}] {"."*int(ii/(len(self.train_gen.dataset)/10))}', end = '\t\t\t\r')

            scheduler.step(total_loss)


            self.recorder['train']['total_loss'].append(total_loss)
            self.recorder['train']['mse'].append(total_loss/len(self.train_gen.dataset))
            print()
            self.eval()
            test_total_loss = 0
            test_yhats = []
            test_ys = []
            jj = 0
            for X,y,resp_selector in self.test_gen:
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
                print(f'Test set... \t mse: [{loss/(X.size(0)):.4f}]   \t |  Epoch progress: [{jj}/{len(self.test_gen.dataset)}] {"."*int(jj/(len(self.test_gen.dataset)/10))}', end = '\t\t\t\r')

            self.recorder['test']['mse'].append(test_total_loss/len(self.test_gen.dataset))

            self.plotter.update(tr_ys=ys[:10000], tr_yhats=yhats[:10000],
                                tst_ys=test_ys[:10000], tst_yhats=test_yhats[:10000],
                                epoch=epoch, tr_loss=self.recorder['train']['mse'][-1],
                                tst_loss=self.recorder['test']['mse'][-1])

            if epoch % PRINT_EVERY == 0:
                print()
                print(f'\t epoch {epoch+1}/{EPOCHS} \t\t\t | \t train total loss: {total_loss:.2f} | train mse: {total_loss/len(self.train_gen.dataset):.6f} || test total_loss: {test_total_loss:.2f} | test mse: {test_total_loss/len(self.test_gen.dataset):.4f}\t\t\t')
        print()
        print('training complete...')
        self.plotter.save_gif(name='test_plot', path='./')
        self.plot_training_loss()

    def plot_training_loss(self):

        plt.figure(figsize=(12,8))
        plt.plot(self.recorder['test']['mse'], 'r-', label='test mse')
        plt.plot(np.linspace(0, len(self.recorder['test']['mse'])-1, len(self.recorder['test']['batch-mse'])),
                    self.recorder['test']['batch-mse'],
                    'r--', label='test batch-mse', alpha=0.75)
        plt.plot(self.recorder['train']['mse'], 'b-', label='train mse')
        plt.plot(np.linspace(0, len(self.recorder['train']['mse'])-1, len(self.recorder['train']['batch-mse'])),
                    self.recorder['train']['batch-mse'],
                    'b--', label='train batch-mse', alpha=0.75)

        plt.xlabel('epochs')
        plt.ylabel('log10(Mean Square Error)')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig('./training_loss_plot.png')

    def unscale_y(self, ys, resp_type):
        '''

        '''
        #return (ys * self.Y_params[resp_type]['std']) + self.Y_params[resp_type]['mean']
        pass














#

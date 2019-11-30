from config import *
import torch
import torch.nn.functional as F
import numpy as np

class Net(torch.nn.Module):

    def __init__(self, train_gen, test_gen):
        super(Net, self).__init__()

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
        loss_func = torch.nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=5, verbose=False, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)  #torch.optim.lr_scheduler.MultiStepLR(optim, milestones=DECAY_MILESTONES, gamma=GAMMA)

        self.recorder={'train':{'total_loss':[], 'mse':[]}, 'val':{'mse':[]}}

        for epoch in range(EPOCHS):
            total_loss = 0
            mse = 0
            yhats=[]
            ys=[]
            self.train()
            ii = 0
            for X, y, resp_type in self.train_gen:
                ii+=1
                X,y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.float)
                yhat = self.forward(X)
                y2 = yhat.detach().numpy().copy()           # only use gradient by one route
                for i, rtype in enumerate(resp_type):      # this could be slow
                    y2[i, RESP_TYPES[rtype]] = y[i]
                y2 = torch.FloatTensor(y2)
                loss = loss_func(yhat, y2)
                optim.zero_grad()
                loss.backward()
                optim.step()
                scheduler.step(loss)
                total_loss += loss.detach().numpy()
                print(f'mse: [{loss.detach().numpy()/(X.size(0)):.5f}]  |  Epoch progress: [{ii*X.size(0)}/{len(self.train_gen.dataset)}] {"."*int(ii/(len(self.train_gen.dataset)/20))}', end = '\r')

                #yhats += yhat.data.numpy().ravel().tolist()
                #ys += y.data.numpy().ravel().tolist()

            self.recorder['train']['total_loss'].append(total_loss)
            self.recorder['train']['mse'].append(total_loss/len(train_gen.dataset))
            if epoch % PRINT_EVERY == 0:
                print(f'epoch {epoch+1}/{EPOCHS} \t|\t train total loss: {total_loss:.2f} \t train mse: {total_loss/len(train_gen.dataset):.2f}\t\t\t', end='\r')
















#

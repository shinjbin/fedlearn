from tkinter import N
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

import torchvision
import matplotlib.pyplot as plt
from .networks import *


class LinearModel(object):
    def __init__(self, net_parameters: dict, lr, train_mode, num_epoch=1, device='cuda'):
        self.device = device
        self.net = LinearNet(net_parameters=net_parameters, train_mode=train_mode).to(device)
        self.num_epoch = num_epoch
        self.lr = lr
        self.train_mode = train_mode
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.RMSprop(params=self.net.parameters(), lr=self.lr)
        

    def train(self, train_data, tol):
        start = time.time()
        self.net.train()

        for epoch in range(self.num_epoch):
            running_loss = 0.0
            prev_loss = 0

            for batch, (x, y) in enumerate(train_data):
                x, y = x.to(self.device), y.to(self.device)
                
                y_hat = self.net.forward(x)

                loss = self.criterion(y_hat, y)
                running_loss += loss.item()

                onehot = F.one_hot(y, num_classes=self.net.num_classes)

                e = torch.sub(y_hat, onehot)

                # backward function
                if self.train_mode == 'dfa':
                    self.net.dfa_backward(e, x)
                elif self.train_mode == 'backprop':
                    loss.backward()
                else:
                    raise Exception("train_mode should be 'dfa' or 'backprop'")
 
                self.optimizer.step()

                if np.abs(loss.item() - prev_loss) <= tol:
                    break
                prev_loss = loss.item()

        time_spent = time.time() - start
        print(f'time spent: {time_spent}')

    def test(self, test_data):
        self.net.to(self.device).eval()
        size = len(test_data.dataset)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for i, (x, y) in enumerate(test_data):
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.net(x)
                test_loss += self.criterion(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            avg_loss = test_loss / size
            accuracy = correct / size

        # print(f'\n Accuracy: {(100 * accuracy):.1f}%, Avg loss: {avg_loss:.8f} \n-------------------------')

        return accuracy

    def example(self, test_data):
        dataiter = iter(test_data)
        x, y = dataiter.next()

        # img_grid = torchvision.utils.make_grid(x, normalize=True)
        # plt.imshow(img_grid.permute(1,2,0))
        # plt.show()
        x = x.to(self.device)
        y = y.to(self.device)

        #ground truth
        print('GroundTruth: ', ' '.join(f'{y[j]}' for j in range(4)))

        #prediction
        self.net.to(self.device)
        with torch.no_grad():
            pred = self.net.forward(x)
            _, predicted = torch.max(pred, 1)
            print('Predicted: ', ' '.join(f'{predicted[j]}' for j in range(4)))

    def save(self):
        torch.save(self.net.state_dict(), self.path)

    def load(self):
        self.net.load_state_dict(torch.load(self.path))
        self.net.to(self.device)
        self.net.eval()

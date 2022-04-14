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


"""Using DFA, one hidden layer, linear layers"""
class OneHiddenNN(nn.Module):
    """
    <default>
    W1 : 800 * 784, W2: 10 * 800
    b1 : 800, b2: 10
    a1 : 4 * 800, a2 : 4 * 10
    y_hat : 4 * 10
    """

    def __init__(self, train_mode='dfa', in_features=784, num_hiddens=800, num_classes=10):
        super().__init__()
        self.in_features = in_features
        self.num_hiddens = num_hiddens
        self.num_classes = num_classes
        self.n = 2
        self.fc1 = nn.Linear(in_features, num_hiddens)
        self.W1 = self.fc1.weight
        self.W1.requires_grad = True
        self.b1 = self.fc1.bias

        self.tanh = F.Tanh()

        self.fc2 = nn.Linear(num_hiddens, num_classes)
        self.W2 = self.fc2.weight
        self.W2.requires_grad = True
        self.b2 = self.fc2.bias
        self.softmax = F.softmax()

        self.gradient_weights = [0, 0]
        self.gradient_biases = [0, 0]

        if train_mode == 'dfa':
            # set parameters zero for dfa
            self.W1.data.fill_(0)
            self.b1.data.fill_(0)
            self.W2.data.fill_(0)
            self.b2.data.fill_(0)

    def forward(self, x):
        x = x.view(-1, self.fc1.in_features)
        self.a1 = self.fc1(x)
        self.h1 = self.tanh(self.a1)
        self.a2 = self.fc2(self.h1)
        self.y_hat = self.softmax(self.a2)

        return self.a1, self.h1, self.a2, self.y_hat

    def dfa_backward(self, e, B1, x):
        x = x.view(-1, self.fc1.in_features)
        dW2 = -torch.matmul(torch.t(e), self.h1)
        da1 = torch.matmul(e, B1) * (1 - torch.tanh(self.a1) ** 2)
        dW1 = -torch.matmul(torch.t(da1), x)
        db1 = -torch.sum(da1, dim=0)
        db2 = -torch.sum(e, dim=0)

        return dW1, dW2, db1, db2

    def backprop_backward(self, e, h1, x):
        x = x.view(-1, self.fc1.in_features)
        # print(f'W2: {self.W2.shape}, W1: {self.W1.shape}, x: {x.shape}, e: {e.shape}, h1: {h1.shape}, a1: {self.a1.shape}')
        dW2 = -torch.matmul(torch.t(e), h1)
        da1 = torch.matmul(e, self.W2) * (1 - torch.tanh(self.a1) ** 2)
        dW1 = -torch.matmul(torch.t(da1), x)
        db1 = -torch.sum(da1, dim=0)
        db2 = -torch.sum(e, dim=0)

        return dW1, dW2, db1, db2

    def parameter_update(self, lr, dW, db):
        with torch.no_grad():
            self.W1 += lr * dW[0]
            self.b1 += lr * db[0]
            self.W2 += lr * dW[1]
            self.b2 += lr * db[1]

    def parameter_renew(self, weights, biases):
        with torch.no_grad():
            self.W1.copy_(weights[0])
            self.b1.copy_(biases[0])
            self.W2.copy_(weights[1])
            self.b2.copy_(biases[1])

    def get_parameters(self):
        return [self.W1, self.W2], [self.b1, self.b2]

    def get_gradients(self):
        return self.gradient_weights, self.gradient_weights


class OneHiddenNNModel(object):
    def __init__(self, lr, train_mode='dfa', criterion=None, num_epoch=10, device='cuda'):
        self.device = device
        self.model = OneHiddenNN(train_mode, 784, 800, 10).to(device)
        self.num_epoch = num_epoch
        self.lr = lr
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()

    def dfa_train(self, B1, train_data, tol=1e-4):

        for epoch in range(self.num_epoch):
            running_loss = 0.0
            prev_loss = 0

            for batch, (x, y) in enumerate(train_data):
                x, y = x.to(self.device), y.to(self.device)

                a1, h1, a2, y_hat = self.model.forward(x)
                onehot = F.one_hot(y, num_classes=10)
                e = torch.sub(y_hat, onehot)

                loss = self.criterion(y_hat, y).item()
                running_loss += loss

                dW1, dW2, db1, db2 = self.model.dfa_backward(e, B1, x)
                self.model.parameter_update(self.lr, [dW1, dW2], [db1, db2])

                if np.abs(loss - prev_loss) <= tol:
                    break
                prev_loss = loss

                gradients_weights, gradients_biases = [dW1, dW2], [db1, db2]
                self.model.gradient_weights = gradients_weights
                self.model.gradient_biases = gradients_biases

                # if batch % 2000 == 1999:
                #     print(f'[{epoch + 1}, {batch + 1:5d}] loss: {running_loss / 2000:.8f}')
                #     running_loss = 0.0

        return [self.model.W1, self.model.W2], [self.model.b1, self.model.b2], gradients_weights, gradients_biases

    def backprop_train(self, train_data, tol=1e-5):
        self.model.to(self.device)

        for epoch in range(self.num_epoch):
            running_loss = 0.0
            prev_loss = 0

            for batch, (x, y) in enumerate(train_data):
                x, y = x.to(self.device), y.to(self.device)

                a1, h1, a2, y_hat = self.model.forward(x)
                onehot = F.one_hot(y, num_classes=10)
                e = torch.sub(y_hat, onehot)

                loss = self.criterion(y_hat, y).item()
                running_loss += loss

                dW1, dW2, db1, db2 = self.model.backprop_backward(e, h1, x)
                self.model.parameter_update(self.lr, [dW1, dW2], [db1, db2])

                gradients_weights, gradients_biases = [dW1, dW2], [db1, db2]

                if np.abs(loss - prev_loss) <= tol:
                    break
                prev_loss = loss

        return [self.model.W1, self.model.W2], [self.model.b1, self.model.b2], gradients_weights, gradients_biases

    def test(self, test_data):
        self.model.to(self.device)
        size = len(test_data.dataset)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for i, (x, y) in enumerate(test_data):
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model.forward(x)[-1]
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
        self.model.to(self.device)
        with torch.no_grad():
            pred = self.model.forward(x)[-1]
            _, predicted = torch.max(pred, 1)
            print('Predicted: ', ' '.join(f'{predicted[j]}' for j in range(4)))

    # def save(self):
    #     torch.save(self.model.state_dict(), self.path)

    # def load(self):
    #     self.model.load_state_dict(torch.load(self.path))
    #     self.model.to(self.device)
    #     self.model.eval()


"""This Class is for linear model with n hidden layers.
    Using DFA or backpropagation.
    activation function: tanh"""
class NHiddenNN(nn.Module):
    """
    <default>
    train_mode: dfa
    W1 : 400 * 784, W2: 400 * 400, W3: 10 * 400
    b1 : 400, b2: 400, b3: 10
    a1 : 4 * 800, a2 : 4 * 10
    y_hat : 4 * 10
    """

    def __init__(self, train_mode, nn_parameters):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.in_features = nn_parameters['in_features']
        self.num_hidden_layer = nn_parameters['num_hidden_layer']
        self.hidden_size = nn_parameters['hidden_size']
        self.num_classes = nn_parameters['num_classes']
        self.batch_size = nn_parameters['batch_size']
        self.n = self.num_hidden_layer + 1
        linear_layer_sizes = [self.in_features] + self.hidden_size
        linear_layer_sizes.append(self.num_classes)

        self.fc = nn.ModuleList()
        for i in range(self.n):
            self.fc.append(nn.Linear(linear_layer_sizes[i], linear_layer_sizes[i+1]))
            
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax()


        """a[i] = W[i]h[i-1] + b[i], h[i] = f(a[i])"""
        self.a = [0] * self.n
        self.h = [0] * (self.n-1)

        # set parameters zero for dfa
        if train_mode == 'dfa':
            for i in range(self.n):
                self.fc[i].weight.data.fill_(0)
                self.fc[i].bias.data.fill_(0)

    def forward(self, x):
        x = x.view(-1, self.in_features)
        self.a[0] = self.fc[0](x)
        self.h[0] = self.activation(self.a[0])
        for i in range(1, self.n-1):
            self.a[i] = self.fc[i](self.h[i-1])
            self.h[i] = self.activation(self.a[i])
        self.a[self.n-1] = self.fc[self.n-1](self.h[self.n-2])
        y_hat = self.softmax(self.a[self.n-1], dim=1)

        return y_hat

    # backward function using DFA(Direct Feedback Alignment)
    def dfa_backward(self, e, B, x):
        da = [0] * (self.n-1)
        x = x.view(-1, self.fc[0].in_features)

        for i in range(self.n-1):
            da[i] = -torch.matmul(e, B[i]) * (1 - torch.tanh(self.a[i]) ** 2)

        for i in range(self.n):
            if i == 0:
                self.fc[i].weight.grad = -torch.matmul(torch.t(da[i]), x)
                self.fc[i].bias.grad = -torch.sum(da[i], dim=0)
            elif i == self.n-1:
                self.fc[i].weight.grad = nn.Parameter(-torch.matmul(torch.t(e), self.h[i-1])) 
                self.fc[i].bias.grad = nn.Parameter(-torch.sum(e, dim=0))
            else:
                self.fc[i].weight.grad = -torch.matmul(torch.t(da[i]), self.h[i-1])
                self.fc[i].bias.grad = -torch.sum(da[i], dim=0)

        

    # backward function using backpropagation
    def backprop_backward(self, e, x):
        da = [0] * (self.n-1)
        x = x.view(-1, self.fc[0].in_features)
        
        for i in reversed(range(self.n)):
            if i == self.n-1:
                da[i-1] = torch.matmul(e, self.fc[i].weight) * (1 - torch.tanh(self.a[i-1]) ** 2)
                self.fc[i].weight.grad = -torch.matmul(torch.t(e), self.h[i-1])
                self.fc[i].bias.grad = -torch.sum(e, dim=0)

            else:
                if i == 0:
                    self.fc[i].weight.grad = -torch.matmul(torch.t(da[0]), x)
                    self.fc[i].bias.grad = -torch.sum(da[0], dim=0)
                else:
                    da[i-1] = torch.matmul(da[i], self.fc[i].weight) * (1 - torch.tanh(self.a[i-1]) ** 2)
                    self.fc[i].weight.grad = -torch.matmul(torch.t(da[i]), self.h[i-1])
                    self.fc[i].bias.grad = -torch.sum(da[i], dim=0)

    def parameter_update(self, lr):
        with torch.no_grad():
            for i in range(self.n):
                self.fc[i].weight += lr * self.fc[i].weight.grad / self.batch_size
                self.fc[i].bias += lr * self.fc[i].bias.grad / self.batch_size

    def parameter_renew(self, weights, biases):
        with torch.no_grad():
            for i in range(self.n):
                self.fc[i].weight.copy_(weights[i])
                self.fc[i].bias.copy_(biases[i])

    def get_parameters(self):
        W = [self.fc[i].weight for i in range(self.n)]
        b = [self.fc[i].bias for i in range(self.n)]
        return W, b

    def get_gradients(self):
        dW = [self.fc[i].weight.grad for i in range(self.n)]
        db = [self.fc[i].bias.grad for i in range(self.n)]
        return dW, db


class NHiddenNNModel(object):
    def __init__(self, nn_parameters, lr, train_mode, num_epoch=10, device='cuda'):
        self.device = device
        self.model = NHiddenNN(nn_parameters=nn_parameters, train_mode=train_mode).to(device)
        self.num_epoch = num_epoch
        self.lr = lr
        self.train_mode = train_mode
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr)
        

    def train(self, B, train_data, tol):
        start = time.time()

        for epoch in range(self.num_epoch):
            running_loss = 0.0
            prev_loss = 0

            for batch, (x, y) in enumerate(train_data):
                x, y = x.to(self.device), y.to(self.device)

                y_hat = self.model.forward(x)

                loss = self.criterion(y_hat, y)
                running_loss += loss.item()

                onehot = F.one_hot(y, num_classes=self.model.num_classes)
                e = torch.sub(y_hat, onehot)
                if self.train_mode == 'dfa':
                    self.model.dfa_backward(e, B, x)
                elif self.train_mode == 'backprop':
                    self.model.backprop_backward(e, x)
                else:
                    raise Exception("train_mode should be 'dfa' or 'backprop'")

                self.model.parameter_update(self.lr)


                if np.abs(loss.item() - prev_loss) <= tol:
                    break
                prev_loss = loss.item()

        time_spent = time.time() - start
        print(f'time_spent: {time_spent}')

    def test(self, test_data):
        self.model.to(self.device)
        size = len(test_data.dataset)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for i, (x, y) in enumerate(test_data):
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model.forward(x)
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
        self.model.to(self.device)
        with torch.no_grad():
            pred = self.model.forward(x)
            _, predicted = torch.max(pred, 1)
            print('Predicted: ', ' '.join(f'{predicted[j]}' for j in range(4)))

    # def save(self):
    #     torch.save(self.model.state_dict(), self.path)

    # def load(self):
    #     self.model.load_state_dict(torch.load(self.path))
    #     self.model.to(self.device)
    #     self.model.eval()
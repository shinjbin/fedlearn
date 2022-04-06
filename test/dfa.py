import matplotlib.pyplot as plt
import numpy as np
import torch.cuda
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy.special import expit # sigmoid
from sklearn.metrics import log_loss
import torch.nn.functional as F


np.random.seed(1234)
batch_size = 4

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
download_root = './MNIST_DATASET'

train_dataset = MNIST(download_root, transform=transform, train=True, download=True)
test_dataset = MNIST(download_root, transform=transform, train=False, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class OneHiddenNN(nn.Module):
    """
    W1 : 800 * 784, W2: 10 * 800
    b1 : 800, b2: 10
    a1 : 4 * 800, a2 : 4 * 10
    y_hat : 4 * 10
    """

    def __init__(self, in_features, num_hiddens, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features, num_hiddens)
        self.W1 = self.fc1.weight
        self.W1.requires_grad = True
        self.b1 = self.fc1.bias
        self.W1.data.fill_(0)
        self.b1.data.fill_(0)

        self.tanh = nn.Tanh()

        self.fc2 = nn.Linear(num_hiddens, num_classes)
        self.W2 = self.fc2.weight
        self.W2.requires_grad = True
        self.b2 = self.fc2.bias
        self.b2.data.fill_(0)
        self.W2.data.fill_(0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        self.a1 = self.fc1(x.view(-1, 784))
        self.h1 = self.tanh(self.a1)
        self.a2 = self.fc2(self.h1)
        self.y_hat = self.sigmoid(self.a2)

        return self.a1, self.h1, self.a2, self.y_hat

    def dfa_backward(self, e, B1, x):
        x = x.view(-1, self.fc1.in_features)
        dW2 = -torch.matmul(torch.t(e), self.h1)
        da1 = torch.matmul(e, B1) * (1 - torch.tanh(self.a1) ** 2)
        dW1 = -torch.matmul(torch.t(da1), x)
        db1 = -torch.sum(da1, dim=0)
        db2 = -torch.sum(e, dim=0)
        return dW1, dW2, db1, db2

    def backprop_backward(self, e, h1, W2, x):
        x = x.view(-1, self.fc1.in_features)
        dW2 = -torch.matmul(torch.t(e), h1)
        da1 = torch.matmul(W2, torch.t(e))*(1-torch.tanh(self.a1)**2)
        dW1 = -torch.matmul(torch.t(da1), x)
        db1 = -torch.sum(da1, dim=0)
        db2 = -torch.sum(e, dim=0)

        return dW1, dW2, db1, db2


    def parameter_update(self, lr, dW1, dW2, db1, db2):
        with torch.no_grad():
            self.W1 += lr * dW1
            self.b1 += lr * db1
            self.W2 += lr * dW2
            self.b2 += lr * db2


class OneHiddenNNModel(object):
    def __init__(self, criterion=None, num_epoch=10, lr=0.001, device='cuda'):
        self.PATH = './MNIST_CLASSIFIER.pth'
        self.device = device
        self.model = OneHiddenNN(784, 800, 10).to(self.device)
        self.num_epoch = num_epoch
        self.lr = lr
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()

    def train(self, train_data, device='cuda', tol=1e-5):

        B1 = torch.randn(10, 800).to(device)

        for epoch in range(self.num_epoch):
            running_loss = 0.0
            prev_loss = 0

            for batch, (x, y) in enumerate(train_data):
                x, y = x.to(device), y.to(device)

                a1, h1, a2, y_hat = self.model.forward(x)
                onehot = F.one_hot(y, num_classes=10)
                e = torch.sub(y_hat, onehot)

                loss = self.criterion(y_hat, y).item()
                running_loss += loss

                dW1, dW2, db1, db2 = self.model.dfa_backward(e, B1, x)
                self.model.parameter_update(self.lr, dW1, dW2, db1, db2)

                if np.abs(loss - prev_loss) <= tol:
                    break

                prev_loss = loss

                if batch % 2000 == 1999:
                    print(f'[{epoch + 1}, {batch + 1:5d}] loss: {running_loss / 2000:.8f}')
                    running_loss = 0.0


    def test(self, test_data, device='cuda'):
        size = len(test_data.dataset)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for i, (x, y) in enumerate(test_data):
                x = x.to(device)
                y = y.to(device)
                pred = self.model.forward(x)[-1]
                test_loss += self.criterion(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        avg_loss = test_loss / size
        accuracy = correct / size

        print(f'\n Accuracy: {(100 * accuracy):.1f}%, Avg loss: {avg_loss:.8f} \n-------------------------')
        return accuracy

    def example(self, test_data):
        dataiter = iter(test_data)
        x, y = dataiter.next()

        img_grid = torchvision.utils.make_grid(x, normalize=True)
        plt.imshow(img_grid.permute(1,2,0))
        plt.show()
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

    def save(self):
        torch.save(self.model.state_dict(), self.PATH)

    def load(self):
        self.model.load_state_dict(torch.load(self.PATH))
        self.model.to(self.device)
        self.model.eval()




model = OneHiddenNNModel()
model.train(train_loader)
model.save()
# model.load()
print(torch.max(model.model.W1), torch.min(model.model.W1))
model.test(test_loader)
model.example(test_loader)


class TwoHiddenNN(nn.Module):
    pass

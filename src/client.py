import gc
import pickle
import logging
import copy
from torch.utils.data import DataLoader
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from .models import OneHiddenNNModel, OneHiddenNN
import torch.utils.data as data
from .ldp_module import LDP


class Client(object):
    def __init__(self, client_id, device='cuda'):
        self.id = client_id
        self.device = device
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        self.nn = None

    def dataload(self, train_dataset, test_dataset, batch_size, path, train_mode='dfa', lr=1e-4):
        self.nn = copy.deepcopy(OneHiddenNNModel(path=path, device=self.device, train_mode=train_mode, lr=lr))

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    def local_train(self, B1, train_mode, tol=1e-4, train=True):
        self.nn.model.to(self.device)
        if train:
            if train_mode == 'dfa':
                W1, W2, b1, b2, gradients = self.nn.dfa_train(train_data=self.train_loader, B1=B1, tol=tol)
            elif train_mode == 'backprop':
                W1, W2, b1, b2, gradients = self.nn.backprop_train(train_data=self.train_loader, tol=tol)
            else:
                raise Exception("train mode is not existing")
            # self.nn.save()
        else:
            self.nn.load()
            W1, W2, b1, b2 = self.nn.model.get_parameters()

        if self.device == "cuda":
            torch.cuda.empty_cache()
        self.nn.model.to("cpu")

        return W1, W2, b1, b2

    def local_eval(self):
        accuracy = self.nn.test(self.test_loader)

        if self.device == "cuda":
            torch.cuda.empty_cache()
        self.nn.model.to("cpu")

        print(f'Accuracy: {accuracy*100:.2f}%')

        return accuracy

    def load_parameters(self):
        pass

    def ldp(self, alpha, c, rho):
        W1 = self.nn.model.W1
        W2 = self.nn.model.W2
        b1 = self.nn.model.b1
        b2 = self.nn.model.b2

        ldp = LDP(W1_user=W1, W2_user=W2, b1_user=b1, b2_user=b2, device=self.device)

        ldp_W1, ldp_W2, ldp_b1, ldp_b2 = ldp.ordinal_cldp(alpha=alpha, c=c, rho=rho)

        return ldp_W1, ldp_W2, ldp_b1, ldp_b2
    
    def gradient_ldp(self, alpha, c, rho):
        gradients = self.nn.model.gradients

        ldp = LDP(W1_user=gradients[0], W2_user=gradients[1], b2_user=gradients[2], device=self.device)

        ldp_dW1, ldp_dW2, ldp_db1, ldp_db2 = ldp.ordinal_cldp(alpha=alpha, c=c, rho=rho)

        return ldp_dW1, ldp_dW2, ldp_db1, ldp_db2




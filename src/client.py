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

    def local_train(self, B1, train_mode, tol, train=True):
        self.nn.model.to(self.device)
        if train:
            if train_mode == 'dfa':
                weights, biases, gradient_weights, gradient_biases = self.nn.dfa_train(train_data=self.train_loader, B1=B1, tol=tol)
            elif train_mode == 'backprop':
                weights, biases, gradient_weights, gradient_biases = self.nn.backprop_train(train_data=self.train_loader, tol=tol)
            else:
                raise Exception("train mode is not existing")
            # self.nn.save()
        else:
            self.nn.load()
            weights, biases = self.nn.model.get_parameters()
            gradient_weights, gradient_biases = 0, 0

        if self.device == "cuda":
            torch.cuda.empty_cache()
        # self.nn.model.to("cpu")

        return weights, biases, gradient_weights, gradient_biases

    def local_eval(self):
        accuracy = self.nn.test(self.test_loader)

        if self.device == "cuda":
            torch.cuda.empty_cache()
        # self.nn.model.to("cpu")

        print(f'Accuracy: {accuracy*100:.2f}%')

        return accuracy

    def load_parameters(self):
        pass

    def ldp(self, alpha, c, rho):
        weights, biases = self.nn.model.get_parameters()

        ldp = LDP(weights=weights, biases=biases, device=self.device)

        ldp_W, ldp_b = ldp.ordinal_cldp(alpha=alpha, c=c, rho=rho)

        return ldp_W, ldp_b
    
    def gradient_ldp(self, alpha, c, rho):
        gradient_weights, gradient_biases = self.nn.model.get_gradients()

        ldp = LDP(weights=gradient_weights, biases=gradient_biases, device=self.device)

        ldp_dW, ldp_db = ldp.ordinal_cldp(alpha=alpha, c=c, rho=rho)

        return ldp_dW, ldp_db




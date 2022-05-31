from re import L
import torch
from torch import nn

"""This Class is for linear model with n hidden layers.
    Using DFA or backpropagation.
    activation function: tanh"""
class LinearNet(nn.Module):
    """
    <default>
    train_mode: dfa
    W1 : 400 * 784, W2: 400 * 400, W3: 10 * 400
    b1 : 400, b2: 400, b3: 10
    a1 : 4 * 800, a2 : 4 * 10
    y_hat : 4 * 10
    """

    def __init__(self, train_mode, net_parameters):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.in_features = net_parameters['in_features']
        self.num_hidden_layer = net_parameters['num_hidden_layer']
        self.hidden_size = net_parameters['hidden_size']
        self.num_classes = net_parameters['num_classes']
        self.batch_size = net_parameters['batch_size']
        self.num_linear_layer = self.num_hidden_layer + 1

        # setup layers
        self.setup_layers()

        # setup DFA. set weights and biases zero / set B
        if train_mode == 'dfa':
            self.setup_dfa()

    def setup_layers(self):
        linear_layer_sizes = [self.in_features] + self.hidden_size
        linear_layer_sizes.append(self.num_classes)

        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.layer_list = []
        for i in range(self.num_linear_layer):
            self.layer_list.append(nn.Linear(linear_layer_sizes[i], linear_layer_sizes[i+1]))
            if i == self.num_linear_layer-1:
                self.layer_list.append(self.softmax)
            else:
                self.layer_list.append(self.activation)
        
        self.sequential = nn.Sequential(*self.layer_list)

        self.a = [0] * self.num_linear_layer
        self.h = [0] * (self.num_linear_layer-1)

        return

    def setup_dfa(self):
        # set weight and bias zero
        for layer in self.layer_list:
            if isinstance(layer, nn.Linear):
                layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)

        self.B = [torch.randn(self.num_classes, self.hidden_size[i]).to(self.device) for i in range(self.num_hidden_layer)]

    def forward(self, x):
        i = 0
        x = x.view(-1, self.in_features)
        for layer in self.layer_list:
            if isinstance(layer, nn.Linear):
                x = layer(x)
                self.a[i] = x

                if i == self.num_linear_layer-1:
                    y_hat = self.softmax(self.a[-1])
                else:
                    x = self.activation(x)
                    self.h[i] = x
                i += 1

        return y_hat

    @staticmethod
    def dtanh(x):
        return 1 - torch.tanh(x)**2

    # backward function using DFA(Direct Feedback Alignment)
    def dfa_backward(self, e, x):
        with torch.no_grad():
            da = [0] * (self.num_linear_layer-1)
            x = x.view(-1, self.in_features)
            

            for i in range(self.num_hidden_layer):
                da[i] = -torch.matmul(e, self.B[i]) * self.dtanh(self.a[i])

            i = 0
            for layer in self.layer_list:
                if isinstance(layer, nn.Linear):
                    if i == 0:
                        layer.weight.grad = torch.matmul(torch.t(da[i]), x) / self.batch_size
                        layer.bias.grad = torch.sum(da[i], dim=0) / self.batch_size
                    elif i == self.num_linear_layer-1:
                        layer.weight.grad = torch.matmul(torch.t(e), self.h[i-1]) / self.batch_size
                        layer.bias.grad = torch.sum(e, dim=0) / self.batch_size
                    else:
                        layer.weight.grad = torch.matmul(torch.t(da[i]), self.h[i-1]) / self.batch_size
                        layer.bias.grad = torch.sum(da[i], dim=0) / self.batch_size
                    i += 1
        

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
        return

    def parameter_update(self, lr):
        with torch.no_grad():
            for layer in self.layer_list:
                if isinstance(layer, nn.Linear):
                    layer.weight += lr * layer.weight.grad
                    layer.bias += lr * layer.bias.grad
        return

    def parameter_copy(self, weights, biases):
        with torch.no_grad():
            i = 0
            for layer in self.layer_list:
                if isinstance(layer, nn.Linear):
                    layer.weight.copy_(weights[i])
                    layer.bias.copy_(biases[i])
                    i += 1
        return

    def get_parameters(self):
        W, b = [], []
        for layer in self.layer_list:
            if isinstance(layer, nn.Linear):
                W.append(layer.weight)
                b.append(layer.bias)
        # W = [layer.weight for layer in self.layer_list if isinstance(layer, nn.Linear)]
        # b = [layer.bias for layer in self.layer_list if isinstance(layer, nn.Linear)]
        return W, b

    def get_gradients(self):
        dW = [layer.weight.grad for layer in self.layer_list if isinstance(layer, nn.Linear)]
        db = [layer.bias.grad for layer in self.layer_list if isinstance(layer, nn.Linear)]
        return dW, db

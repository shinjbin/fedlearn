import torch
import copy

from .models import NHiddenNNModel, NHiddenNN
from .client import Client

"""This is Class for aggregation module.
    this performs creating clients, 
    train/test clients, 
    averaging parameter/gradients of clients,
    send parameters to local models of clients"""
class Aggregation(object):
    def __init__(self, device, train_mode, nn_parameters):
        self.device = device
        self.clients = []
        self.num_client = 0
        self.global_model = NHiddenNNModel(device=device, lr=0.001, train_mode=train_mode, nn_parameters=nn_parameters)
        self.n = self.global_model.model.n
        self.nn_parameters = nn_parameters

    def create_clients(self, client_id):
        client = copy.deepcopy(Client(client_id=client_id, device=self.device))
        self.clients.append(client)
        self.num_client += 1

    def train_client(self, train_mode, tol, hidden_size, num_classes):
        B = [0] * (self.n-1)
        for i in range(self.n-1):
            B[i] = torch.randn(num_classes, hidden_size[i]).to(self.device)

        for i in range(self.num_client):
            print(f'Training client {i}...')
            self.clients[i].local_train(B=B, tol=tol)

    def test_client(self):
        for i in range(self.num_client):
            print(f"Client {i} testing...")
            self.clients[i].local_eval()

    def parameter_client(self, index):
        W1, W2, b1, b2 = self.clients[index].nn.model.get_parameters()
        return W1, W2, b1, b2

    # averaging gradients in aggregation module
    def average_gradients(self, alpha, c, rho, ldp):
        avg_dW1, avg_dW2, avg_db1, avg_db2 = 0, 0, 0, 0
        for i in range(self.num_client):
            ldp_dW1, ldp_dW2, ldp_db1, ldp_db2 = self.clients[i].gradient_ldp(alpha=alpha, c=c, rho=rho)
                
            avg_dW1 += ldp_dW1
            avg_dW2 += ldp_dW2
            avg_db1 += ldp_db1
            avg_db2 += ldp_db2

        print("--------------------------------")
        print('parameter averaging...')
        
        avg_dW1 /= self.num_client
        avg_dW2 /= self.num_client
        avg_db1 /= self.num_client
        avg_db2 /= self.num_client
        

        return avg_dW1, avg_dW2, avg_db1, avg_db2

    # averaging parameters(weights, biases) in aggregation module
    def average_parameters(self, alpha, c, rho, ldp):
        avg_W = [0] * self.n
        avg_b = [0] * self.n
        if ldp:
            for i in range(self.num_client):
                ldp_W, ldp_b = self.clients[i].ldp(alpha=alpha, c=c, rho=rho)

                for i in range(self.n):
                    avg_W[i] += ldp_W[i]
                    avg_b[i] += ldp_b[i]

        else:
            for i in range(self.num_client):
                temp_W, temp_b = self.clients[i].nn.model.W, self.clients[i].nn.model.b
                for w in temp_W:
                    w.to(self.device)
                for b in temp_b:
                    b.to(self.device)
                
                for i in range(self.n):
                    avg_W[i] += temp_W[i]
                    avg_b[i] += temp_b[i]
        
        print("--------------------------------")
        print('parameter averaging...')
        
        for i in range(self.n):
            avg_W[i] /= self.num_client
            avg_b[i] /= self.num_client

        return avg_W, avg_b

            # check_w1 = torch.equal(ldp_W1, temp_W1)
            # check_w2 = torch.equal(ldp_W1, temp_W1)
            # check_b1 = torch.equal(ldp_W1, temp_W1)
            # check_b2 = torch.equal(ldp_W1, temp_W1)
            # print(check_w1, check_w2, check_b1, check_b2)

    def global_gradient_update(self, alpha, c, rho, ldp):
        gradient_weights, gradient_biases = self.average_gradients(alpha=alpha, c=c, rho=rho, ldp=ldp)

        self.global_model.parameter_update(gradient_weights, gradient_biases)

    def global_parameter_update(self, alpha, c, rho, ldp):
       
        weights, biases = self.average_parameters(alpha=alpha, c=c, rho=rho, ldp=ldp)

        self.global_model.model.parameter_renew(weights, biases)
    

    def local_parameter_update(self):
        print('local model parameter updating...')
        weights, biases = self.global_model.model.get_parameters()

        for i in range(self.num_client):
            self.clients[i].nn.model.parameter_renew(weights, biases)

        print("--------------------------------")




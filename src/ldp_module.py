import numpy as np
import random
import torch
import copy
import torch.multiprocessing as mp

"""class for LDP module.
input: weights, bias, device(cpu or cuda)
it performs alpha-CLDP and output perturbed data."""
class LDP(object):
    def __init__(self, weights, biases, device):
        self.W = []
        self.b = []
        for weight in weights:
            self.W.append(copy.deepcopy(weight).to(device))
        
        for bias in biases:
            self.b.append(copy.deepcopy(bias).to(device))
        self.device = device

    @staticmethod
    def perturb_weight(W, alpha, u):
        # a = torch.empty()
        # for v in self.u:
        #     a.append(torch.full((W.shape[0], W.shape[1]), v))

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):

                d = torch.abs(torch.sub(u, W[i][j]))
                score = torch.exp(-alpha * d / 2)
                # prob = []
                # sum_sc = torch.sum(score).to('cuda')
                # for s in score:
                #     prob.append(s / sum_sc)
                # prob = torch.tensor(prob)

                idx = score.multinomial(num_samples=1, replacement=True)
                with torch.no_grad():
                    W[i][j] = u[idx]
        return W
    
    @staticmethod
    def perturb_bias(b, alpha, u):
        for i in range(b.shape[0]):
            d = torch.abs(torch.sub(u, b[i]))
            score = torch.exp(-alpha * d / 2)
            # prob = []
            # sum_sc = torch.sum(score)
            # for s in score:
            #     prob.append(s / sum_sc)
            # prob = torch.tensor(prob)

            idx = score.multinomial(num_samples=1, replacement=True)
            with torch.no_grad():
                b[i] = u[idx]
        return b

    """performing ordinal cldp.
    input: alpha, c, rho,
    output: W1,W2,b1,b2"""
    def ordinal_cldp(self, alpha, c, rho):
        print("ordinal CLDP running...")

        # u: item universe
        u = torch.arange(-c*(10**rho), c*(10**rho)+1).to(self.device)

        with torch.no_grad():
            for i in range(len(self.W)):
                self.W[i] *= 10**rho
                self.W[i] = self.perturb_weight(self.W[i], alpha=alpha, u=u)
                self.W[i] /= 10**rho
            for i in range(len(self.b)):
                self.b[i] *= 10**rho
                self.b[i] = self.perturb_bias(self.b[i], alpha=alpha, u=u)
                self.b[i] /= 10**rho
        
        
        # self.W1 = self.perturb_weight(self.W1, alpha=alpha, u=u)
        # self.W2 = self.perturb_weight(self.W2, alpha=alpha, u=u)
        # self.b1 = self.perturb_bias(self.b1, alpha=alpha, u=u)
        # self.b2 = self.perturb_bias(self.b2, alpha=alpha, u=u)

        # self.W1 /= 10**rho
        # self.W2 /= 10**rho
        # self.b1 /= 10**rho
        # self.b2 /= 10**rho

        return self.W, self.b

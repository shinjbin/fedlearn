import os
import pandas as pd
import csv
from csv import writer
import time

"""Wrtie csv file"""
class Utils(object):
    def __init__(self, **kargs):
        self.device = kargs['device']
        self.train_mode = kargs['train_mode']
        self.dt = kargs['dt']
        self.num_clients = kargs['num_clients']
        self.batch_size = kargs['batch_size']
        self.num_round = kargs['num_round']
        self.learning_rate = kargs['learning_rate']
        self.tol = kargs['tol']
        self.ldp = kargs['ldp']
        self.alpha = kargs['alpha']
        self.c = kargs['c']
        self.rho = kargs['rho']
        self.global_accuracy = kargs['global_accuracy']
        self.start_time = kargs['start_time']
        self.hidden_size = kargs['hidden_size']

    def print_info(self):
            print(f'--------------------------------------<{self.train_mode}>---------------------------------------------\n'
                f'starting datetime: {self.dt}\n'
                f'num_clients = {self.num_clients} batch_size = {self.batch_size}, num_round = {self.num_round}, '
                f'learning_rate = {self.learning_rate}, tol = {self.tol}, ldp = {self.ldp}, alpha = {self.alpha}\n'
                f'MODEL: num_hidden_layer = {self.num_hidden_layer}, in_features = {self.in_features}, '
                f'hidden_size = {self.hidden_size}, '
                f'output_size = {self.num_classes}\n'
                f'---------------------------------------------------------------------------------------------\n\n')
        

    def write_csv(self):

        if os.path.isfile('result.csv'):
            list_data = [self.train_mode, self.dt, self.num_clients, self.batch_size, self.num_round, self.learning_rate,
                        self.tol, self.ldp, self.alpha, self.c, self.rho, f'{self.global_accuracy*100:.2f}%',
                        f'{time.time()-self.start_time:.2f}', self.hidden_size]
            with open('result.csv', 'a', newline='') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(list_data)
                f_object.close()
                
        else:
            columns = ['train mode', 'starting datetime', 'num_clients', 'batch_size', 'num_round', 'learning_rate',
                        'tol', 'ldp', 'alpha', 'c', 'rho', 'Final global model accuracy', 'time taken']
            dict_df = {'train mode': self.train_mode, \
                                'starting datetime': self.dt, \
                                'num_clients': self.num_clients, \
                                'batch_size': self.batch_size, \
                                'num_round': self.num_round, \
                                'learning_rate': self.learning_rate, \
                                'tol': self.tol, \
                                'ldp': self.ldp, \
                                'alpha': self.alpha, \
                                'c': self.c, \
                                'rho': self.rho, \
                                'Final global model accuracy': f'{self.global_accuracy*100:.2f}%', \
                                'time taken': f'{time.time()-self.start_time:.2f}'}
            df = pd.DataFrame([dict_df], columns=columns)
            df.to_csv('result.csv', index=False)
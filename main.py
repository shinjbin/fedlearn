import random
import time
import datetime
import threading
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.utils.data import DataLoader
import os
import pandas as pd
from csv import writer
import csv

from src.dataset import Dataset
from src.aggregation import Aggregation


def k_selection(parameters_list, k):
    return random.choices(parameters_list, k=k)


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    print('------------------------------------------')

    # hyperparameter
    num_clients = 5
    batch_size = 4
    num_round = 5
    train_mode = 'dfa'  # 'dfa' or 'backprop'
    learning_rate = 0.001
    tol = 0.0005
    

    # ldp parameters
    ldp = 'gradient' # 'gradient' or 'parameter' or False
    if ldp:
        alpha = 0.1
        c = 1
        rho = 3
    else:
        alpha = False
        c = False
        rho = False

    for stage in range(10):
        print('current stage:', stage)
        dt = datetime.datetime.now()
        start_time = time.time()


        # load datasets and split
        dataset = Dataset()
        train_dataset_split, test_dataset = dataset.split(num_clients)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        path_list = []
        for i in range(num_clients):
            path_list.append(f'CLIENT{i}_MNIST_CLASSIFIER.pth')
        global_path = 'GLOBAL_MNIST_CLASSIFIER.pth'

        # create aggregation module
        aggregation = Aggregation(device, global_path, train_mode)

        # create clients
        clients = []
        print(f'Creating {num_clients} clients...')
        for i in range(num_clients):
            aggregation.create_clients(i, alpha=alpha)
            aggregation.clients[i].dataload(train_dataset=train_dataset_split[i],
                                            test_dataset=test_dataset,
                                            batch_size=batch_size,
                                            path=path_list[i],
                                            train_mode=train_mode,
                                            lr=learning_rate)

        # federated learning start
        for r in range(num_round):
            print(f'---------------<Round {r}>----------------')

            # train clients local model
            aggregation.train_client(train_mode=train_mode, tol=tol)
            print("--------------------------------")

            # test clients local model
            aggregation.test_client()
            print("--------------------------------")

            # performing alpha-CLDP and update global model parameter(parameter averaging)
            print(f'ldp parameters: alpha:{alpha:.2f}, c:{c}, rho:{rho}')
            if ldp == 'gradient':
                aggregation.global_gradient_update(alpha=alpha, c=c, rho=rho, ldp=ldp)
            else:
                aggregation.global_parameter_update(alpha=alpha, c=c, rho=rho, ldp=ldp)

            # test global model
            global_accuracy = aggregation.global_model.test(test_loader)
            print(f'Global model accuracy: {global_accuracy*100:.2f}%')
            print("--------------------------------")
        

            # local model parameter update
            aggregation.local_parameter_update()
        

        global_accuracy = aggregation.global_model.test(test_loader)
        aggregation.global_model.example(test_loader)

        # write result file
        f = open("result.txt", "a")
        f.write(f'--------------------------------------<{train_mode}>---------------------------------------------\n'
                f'starting datetime: {dt}\n'
                f'num_clients = {num_clients} batch_size = {batch_size}, num_round = {num_round}, '
                f'learning_rate = {learning_rate}, tol = {tol}, ldp = {ldp}, alpha = {alpha}\n\n'
                f'Final global model accuracy using {train_mode}: {global_accuracy*100:.2f}%\n'
                f'time taken using {train_mode}: {time.time()-start_time:.2f}\n'
                f'---------------------------------------------------------------------------------------------\n\n')
        f.close()

        global_accuracy_df = round(global_accuracy*100, 2)
        time_spent = round(time.time()-start_time, 2)

        # write csv file
        if os.path.isfile('result.csv'):
            list_data = [train_mode, dt, num_clients, batch_size, num_round, learning_rate,
                    tol, ldp, alpha, c, rho, f'{global_accuracy*100:.2f}%', f'{time.time()-start_time:.2f}']
            with open('result.csv', 'a', newline='') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(list_data)
                f_object.close()
    
        else:
            columns = ['train mode', 'starting datetime', 'num_clients', 'batch_size', 'num_round', 'learning_rate',
            'tol', 'ldp', 'alpha', 'c', 'rho', 'Final global model accuracy', 'time taken']
            dict_df = {'train mode': train_mode, \
                    'starting datetime': dt, \
                    'num_clients': num_clients, \
                    'batch_size': batch_size, \
                    'num_round': num_round, \
                    'learning_rate': learning_rate, \
                    'tol': tol, \
                    'ldp': ldp, \
                    'alpha': alpha, \
                    'c': c, \
                    'rho': rho, \
                    'Final global model accuracy': global_accuracy_df, \
                    'time taken': time_spent}
            df = pd.DataFrame([dict_df], columns=columns)
            df.to_csv('result.csv', index=False)


        torch.cuda.empty_cache()

        alpha /= 1.5

        


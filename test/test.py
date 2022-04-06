import torch
import time

start = time.time()

a = torch.ones(4,3,2)
b = torch.ones(4,3)
print(a, b)
print(a-b)


total_time = time.time() - start
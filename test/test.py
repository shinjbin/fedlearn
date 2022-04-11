from tkinter import N
import torch
import time

# start = time.time()

# a = torch.ones(4,3,2)
# b = torch.ones(4,3)
# print(a, b)
# print(a-b)


# total_time = time.time() - start

# class A(object):
#     def __init__(self, num):
#         self.num = num
    
#     def plus(self):
#         self.num += 1

# num = 1
# a = A(num)

# a.plus()

# print(num)

a = [0] * 3

a[0] = torch.rand(2,3)
a[1] = torch.rand(3,2)

print(a)
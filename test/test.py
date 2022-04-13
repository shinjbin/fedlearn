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

shape = (3,2,3)
a = torch.rand(shape)

b = torch.rand((2,3))
c = torch.rand((2,3))
d = torch.rand((2,3))

a[0] = b
a[1] = c
a[2] = d
print(a)
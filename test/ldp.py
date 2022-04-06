import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd
import numpy as np

adult = pd.read_csv("../adult_with_pii.csv")
def laplace_mech(v, sensitivity, epsilon):
    return v + np.random.laplace(loc=0, scale=sensitivity / epsilon)
def pct_error(orig, priv):
    return np.abs(orig - priv)/orig * 100.0


# Randomized Response
def rand_resp_sales(response):
    truthful_response = response == 'Sales'

    # first coin flip
    if np.random.randint(0, 2) == 0:
        # answer truthfully
        return truthful_response
    else:
        # answer randomly (second coin flip)
        return np.random.randint(0, 2) == 0


# responses = [rand_resp_sales(r) for r in adult['Occupation']]
# print(len(adult[adult['Occupation'] == 'Sales']))
#
# # we expect 1/4 of the responses to be "yes" based entirely on the coin flip
# # these are "fake" yesses
# fake_yesses = len(responses)/4
#
# # the total number of yesses recorded
# num_yesses = np.sum([1 if r else 0 for r in responses])
#
# # the number of "real" yesses is the total number of yesses minus the fake yesses
# true_yesses = num_yesses - fake_yesses
#
# rr_result = true_yesses*2
# print(rr_result)


# Unary Encoding
domain = adult['Occupation'].dropna().unique()

def encode(response):
    return [1 if d == response else 0 for d in domain]

def perturb(encoded_response):
    return [perturb_bit(b) for b in encoded_response]

def perturb_bit(bit):
    p = .75
    q = .25

    sample = np.random.random()
    if bit == 1:
        if sample <= p:
            return 1
        else:
            return 0
    elif bit == 0:
        if sample <= q:
            return 1
        else:
            return 0

def unary_epsilon(p, q):
    return np.log((p*(1-q)) / ((1-p)*q))

counts = np.sum([encode(r) for r in adult['Occupation']], axis=0)
print(list(zip(domain, counts)))

counts = np.sum([perturb(encode(r)) for r in adult['Occupation']], axis=0)
print(list(zip(domain, counts)))

def aggregate(responses):
    p = .75
    q = .25

    sums = np.sum(responses, axis=0)
    n = len(responses)

    return [(v - n * q) / (p - q) for v in sums]

responses = [perturb(encode(r)) for r in adult['Occupation']]
counts = aggregate(responses)
print(list(zip(domain, counts)))
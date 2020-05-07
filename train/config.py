import torch

'''
Training
'''

# Sanity Check
sanity = False
subset = 10

# Learning parameters
weights = torch.tensor([0.05, 0.95]) # background, road
lr = 1e-3
summary_prefix = "shellnet"
num_epoch = 50
batch_size = 1
reg_factor = 0

# Loading dicts
load = False
model = None

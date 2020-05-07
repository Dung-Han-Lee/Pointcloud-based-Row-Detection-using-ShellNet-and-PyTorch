import torch

'''
Training
'''

# Sanity Check
sanity = False
subset = 10

# Learning parameters
weights = torch.tensor([0.1, 0.9]) # background, road
lr = 3e-4
summary_prefix = "shellnet"
num_epoch = 100
batch_size = 32
reg_factor = 0
fc_scale = 1
conv_scale = 1

# Loading dicts
load = False
model ="/42.pth"

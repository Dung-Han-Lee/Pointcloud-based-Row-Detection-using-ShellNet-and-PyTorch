
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import  WeightedRandomSampler
from sys import argv

#User define modules
import data
import model
import routine
import config

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cross_entropy(inputs, target, weight=None, size_average=True):
    n, h, c = inputs.size()

    # CHW -> HWC -> (HW) x c
    inputs = inputs.contiguous().view(-1, c)
    target = target.view(-1)    
    loss = F.cross_entropy(inputs, target, weight=weight, 
        size_average=size_average)

    return loss
    
def run():
    train_dataset = data.PointcloudDataset(mode = 'train')
    val_dataset   = data.PointcloudDataset(mode = 'val')
        
    train_loader = DataLoader( train_dataset, 
                               batch_size=config.batch_size,
                               shuffle=True, 
                               drop_last = False)

    val_loader = DataLoader( val_dataset, 
                             batch_size=config.batch_size, 
                             shuffle=False, 
                             drop_last = False)
    
    # Set up models and (optionally) load weights
    num_class = 2
    num_points = 1024
    network = model.ShellNet(num_class, num_points)
    if config.load:
        state_dict = torch.load('./models/'+config.model)
        network.load_state_dict(state_dict)
        print("loading weights from {}".format(config.model) )

    if not config.load:
        print("train from scracth...")

    network.train()
    network.to(device)
            
    # Define loss function then start to train
    criterion = cross_entropy
    optimizer = torch.optim.Adam(network.parameters(), lr = config.lr)
    routine.train(network, train_loader, val_loader, criterion, optimizer)

if __name__ == "__main__":
    run()


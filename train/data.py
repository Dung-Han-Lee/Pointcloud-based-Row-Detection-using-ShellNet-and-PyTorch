
import os
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
import scipy.misc as m
from math import pi, cos, sin

#User define modules
import paths
import config

my_path = os.path.abspath(os.path.dirname(__file__))

class PointcloudDataset(data.Dataset):
    def __init__(self, mode):
        assert mode == 'train' or mode =='val' or mode =='test', \
                "mode must be one of  the following: train, val, test"
        self.mode = mode
        self.pc_base  = os.path.join(my_path, paths.base, mode, "pointcloud")
        self.lbl_base = os.path.join(my_path, paths.base, mode, "pc_label")

        print("during {}, # of inputs = {}, labels = {}".format(\
            mode, len(os.listdir(self.pc_base)), len(os.listdir(self.lbl_base)) ))
        assert len(os.listdir(self.pc_base))==len(os.listdir(self.lbl_base))
        
    def __len__(self):
        if config.sanity:
            return config.subset
        return len(os.listdir(self.pc_base)) 

    def __getitem__(self, index):
        pc_path = os.path.join(self.pc_base, ("%04d" % index ) + ".npy")
        lbl_path = os.path.join(self.lbl_base, ("%04d" % index ) + ".npy")
        pc  = np.load( pc_path, allow_pickle=True)
        lbl = np.load(lbl_path, allow_pickle=True)
        if self.mode == "train":
            pc = self.transform(pc)

        return pc, lbl
 
    def transform(self, pointcloud):
        ang = np.random.uniform(-pi, pi)
        rot_z = np.array([  [cos(ang), -sin(ang), 0],
                            [sin(ang),  cos(ang), 0],
                            [       0,         0, 1]])
        return pointcloud @ rot_z
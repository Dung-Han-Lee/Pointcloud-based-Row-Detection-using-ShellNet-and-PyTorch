

import sys
import pdb
import argparse
from math import pi, radians, degrees, sin, cos
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.append("../train")
import model
import data
import paths
sys.path.append("../utils")
from viz_pointcloud import show_semantic

class Test:
    def __init__(self, path_weight):
        self.device = torch.device('cuda' if \
            torch.cuda.is_available() else 'cpu') 
        
        # Initialize model
        weight = torch.load(path_weight, map_location = self.device)
        self.network = model.ShellNet(2, 1024)
        self.network.load_state_dict(weight)
        self.network.eval()
        self.network.to(self.device)

    def evaluate(self, test_loader):

        angle = radians(-30)
        rot_z = np.array([  [cos(angle), -sin(angle), 0],
                            [sin(angle),  cos(angle), 0],
                            [         0,           0, 1]])

        for num, (pointcloud, labels) in enumerate(test_loader):
            (pointcloud, labels) = map(lambda x : x.to(self.device), (pointcloud, labels))
            output = self.network(pointcloud.float()).detach().numpy() 
            #output_class = np.argmax(output, axis=2).flatten()
            #pc = pointcloud.squeeze(0).detach().numpy() @ rot_z
            #show_semantic(output_class, pc, view='top')

            import pdb
            pdb.set_trace()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evalutate results of range image unet.')
    parser.add_argument("-m", "--model", required=True, nargs="+",  help="paths to deep learning segmentation model")
    args = vars(parser.parse_args())

    test_dataset  = data.PointcloudDataset(mode = 'test')
    test_loader   = DataLoader(test_dataset, batch_size=2)
    
    for path in (args["model"]):
        print(path)
        test = Test(path) #best 0406_RIU03_60
        test.evaluate(test_loader)
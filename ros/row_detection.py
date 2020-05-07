# Torch
import torch

# Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Ros
import rospy
from numpy_pc2 import pointcloud2_to_xyzi_array
from sensor_msgs.msg import PointCloud2
from time import sleep

import sys
sys.path.append("../train")
import model
sys.path.append("../utils")
from viz_pointcloud import show_semantic

class Node:
    def __init__(self, path_weight):
        self.pc = None
        self.device = torch.device('cuda' if \
            torch.cuda.is_available() else 'cpu') 
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim3d(0,20)
        self.ax.set_ylim3d(-10,10)
        self.ax.set_zlim3d(-10,10)
        self.ax.view_init(60,-180)

        # Initialize model
        weight = torch.load(path_weight, map_location = self.device)
        self.network = model.ShellNet(2, 1024, conv_scale=2,  dense_scale=2)
        self.network.load_state_dict(weight)
        self.network.eval()
        self.network.to(self.device)

    def pc_callback(self, pc):
        pc = pointcloud2_to_xyzi_array(pc) 
        idx = np.random.permutation(len(pc))[:1024]
        self.pc = pc[idx]
        

    def visualize(self):
        if self.pc is None:
            return
        pc = self.pc.copy()
        inputs = torch.from_numpy(pc[:, :3]).unsqueeze(0).float().to(self.device)
                
        output = self.network(inputs).detach().numpy()
        output_class = np.argmax(output, axis=2).flatten()
        pc_row = pc[output_class == 1, :]
        pc_else = pc[output_class == 0, :]
        pts1 = self.ax.scatter( pc_row[:, 0], pc_row[:, 1] , pc_row[:, 2], c= 'r', marker='o')
        pts2 = self.ax.scatter( pc_else[:, 0], pc_else[:, 1] , pc_else[:, 2], c= 'b', marker='o')
        plt.pause(0.01)
        pts1.remove()
        pts2.remove()


if __name__ == '__main__':

    rospy.init_node('row_dectection')
    rate = rospy.Rate(20) 
    
    node = Node("../weights/shellnet01/91.pth")
 
    # subscribers
    rospy.Subscriber("velodyne_points", PointCloud2, node.pc_callback)   

    while not rospy.is_shutdown():

        # update subscribers        
        sleep(0.1)
        node.visualize()
       
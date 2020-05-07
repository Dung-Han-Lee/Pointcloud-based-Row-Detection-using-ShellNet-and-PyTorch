from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
import numpy as np
from math import pi, radians, degrees, cos, sin
import sys
import cv2
import os

def show_semantic(label, pointcloud, view='front', color='r'):
    """
    Args:
        label      : (N, ) numpy semantic annotation of target of two classes [0, 1]
        pointcloud : (N,4) numpy pointclouds [x, y, z]
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Draw class 0
    pc1 = pointcloud[label==0, :]
    ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], s=1, c='b', marker='o', alpha=0.5)

    # Draw class 1
    pc2 = pointcloud[label==1, :]
    ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], s=3, c=color, marker='o')

    print("pc1 = {}, pc2 = {}, pointcloud = {}".format(pc1.shape, pc2.shape, pointcloud.shape))

    for xb, yb, zb in zip(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2]):
       ax.plot([xb], [yb], [zb], 'w')

    if view == 'front':
        ax.view_init(0, -180)
        plt.title('front')
    elif view == 'top':
        ax.view_init(90,-180)
        plt.title('top')
    elif view == 'side':
        ax.view_init(0, 90)
        plt.title('side')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim3d(0,20)
    ax.set_ylim3d(-10,10)
    ax.set_zlim3d(-10,10)
    plt.show()
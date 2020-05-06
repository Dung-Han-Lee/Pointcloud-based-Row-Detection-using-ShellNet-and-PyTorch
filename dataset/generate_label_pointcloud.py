from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
import numpy as np
from math import pi, radians, degrees, cos, sin
import sys
import cv2
import os
import tkinter as tk

SCALE = 10
DEMO_IMAGE_WIDTH  = 500
DEMO_IMAGE_HEIGHT = 500
VELODYNE_VERTICAL_RES = radians(2)
VELODYNE_VERTICAL_MIN_ANGLE = radians(-15)
VELODYNE_NUM_BEAMS = 16

def show_semantic(label, pointcloud, view='front'):
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
    ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], s=3, c='r', marker='o')

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
    #plt.show()

def register_label(label_img):
    h, w = label_img.shape
    compressed = np.zeros((int(h/4), w))
    for i in range(h):
        if(i%4==0):
            compressed[int(i/4)] = label_img[i]
    label_img = compressed
    return np.fliplr(label_img)

def get_row_indices(pointcloud, label_img, theta_min=-pi/2, theta_max=pi/2):
    """
    Args:
        pointcloud: 
            Nx4 numpy array of 3D point coordinates in lidar sensor frame
            the corresponding fields are x,y,z,intensity
        theta_max/min:  
            the allowed horizontal theta range with forward defined as 0, 
            left as pi/2 (due to arctan2 property) 

    Returns:
        Range view image

    Notes: 
        z is align with rotation axis pointing outward from the top of device
        x pointing forward (due to ROS convention) 
    """
    
    # Convert cartesian to spherical
    r = np.linalg.norm(pointcloud[:,0:3], axis=1)
    phi = np.arccos(pointcloud[:,2] / r) - (pi / 2) # top: -pi/2, down: pi/2
    theta = np.arctan2(pointcloud[:,1], pointcloud[:,0])

    # Calculate vetical phi with top beam corrs to index 0, btm index 15
    y = np.rint((phi - VELODYNE_VERTICAL_MIN_ANGLE) * (1./VELODYNE_VERTICAL_RES)).astype('int')

    # Calculate horizontal theta with right corrs to index 0 (need flip later)
    theta_step = radians(0.35)
    x_len = 512
    x = np.rint((theta - theta_min) * (1./theta_step)).astype('int')

    # Force x range within [0, 512]
    # Since row is always in center, these points would not be labeled as row
    x[ x < 0 ] = 0
    x[ x >= x_len] = 0

    valid = label_img[y, x] == 255
    return valid



if __name__ == '__main__':
    root = "."
    src_folder_names = ['train', 'val', 'test']

    for folder_name in src_folder_names:
        npz_folder = os.path.join(root, folder_name, "npz")
        lbl_folder = os.path.join(root, folder_name, "label")
        out_pc_folder = os.path.join(root, folder_name, "pointcloud")
        out_lbl_folder = os.path.join(root, folder_name, "pc_label")
        for i, npz_name in enumerate(sorted(os.listdir(npz_folder))):
            pc = np.load(os.path.join(npz_folder, npz_name), allow_pickle=True)["pointcloud"][:, :3]
            lbl = cv2.imread(os.path.join(lbl_folder, sorted(os.listdir(lbl_folder))[i]), 0)
            lbl = register_label(lbl)

            label_prefix = sorted(os.listdir(lbl_folder))[i][:-len("_visualize.png")]
            npz_prefix = npz_name[:-len(".npz")]
            #print("label prefix = {} must match npz prefix {}".format(label_prefix, npz_prefix))
            assert label_prefix == npz_prefix, "label prefix must match npz prefix"

            label = get_row_indices(pc, lbl)
            
            # Subsample 1025 points
            idx = np.random.permutation(len(pc))[:1024]
            (pc, label) = map(lambda x : x[idx], (pc, label))

            np.save(os.path.join(out_pc_folder, str(npz_prefix)) , pc)
            np.save(os.path.join(out_lbl_folder, str(label_prefix)), label)

            if i % 10 == 9:
                show_semantic(label, pc, view='top')
                plt.show()
            


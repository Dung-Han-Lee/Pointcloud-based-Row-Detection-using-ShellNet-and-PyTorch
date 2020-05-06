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
VELODYNE_NUM_BEAMS = 16
VELODYNE_VERTICAL_MAX_ANGLE = radians(15)

def topview(pointcloud):
    """
    Args:
        pointcloud: 
            N x D (D>=3), numpy array of 3D point coordinates in lidar sensor frame
            the corresponding fields are x,y,z,intensity etc
    Returns:
        image with following correspondance:
            x (forward in sensor frame) -> row
            y (horizon in sensor frame) -> col
    """
    img = np.zeros((DEMO_IMAGE_HEIGHT, DEMO_IMAGE_WIDTH))
    (x, y, z) = (pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2])

    # Scale values (c=0 corres to center line)
    c = SCALE * y + DEMO_IMAGE_WIDTH/2
    r = x * SCALE
    z = 255 * (z - np.min(z))/(np.max(z) - np.min(z))

    # Mask out invalid points
    valid = (r >= 0) & (r < DEMO_IMAGE_HEIGHT - 1) &\
            (c >= 0) & (c < DEMO_IMAGE_WIDTH  - 1)

    r = np.rint(r).astype('int')
    c = np.rint(c).astype('int')
    rvalid = r[valid]
    cvalid = c[valid]
    zvalid = z[valid]

    img[rvalid, cvalid] = zvalid

    # Flip the image so r=0 is at btm, c=0 at right
    img = np.fliplr(img)
    img = np.flipud(img)

    return img

def get_tree_indices(pointcloud, topview):
    """
    Args:
        pointcloud  (N X 3 numpy float64) corresponding to label pixels (x, y, z)
        topview     (H x W numpy float64)  
    Returns:
        indices     (M, numpy) indices corresponding to trees 
    """
    topview = np.fliplr(np.flipud(topview))
    res = []
    for point in pointcloud:
        x, y, z = point
        c = SCALE * y + DEMO_IMAGE_WIDTH/2
        r = x * SCALE
        (r, c) = map(lambda x : np.rint(x).astype('int'), (r, c))
        if r < 0 or r >= DEMO_IMAGE_HEIGHT or c < 0 or c >= DEMO_IMAGE_WIDTH:
            v = 0
        else:    
            v = 1 if (topview[r, c] > 200) else 0
        res.append(v)

    return np.array(res)


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

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/yrange9/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def EvaluateLabel(rot_y):
    root = "."
    src_folder_names = ['train', 'val', 'test']

    for folder_name in src_folder_names:
        npz_folder = os.path.join(root, folder_name, "npz")
        lbl_folder = os.path.join(root, folder_name, "pc_label")
        for i, npz_name in enumerate(sorted(os.listdir(npz_folder))):

            print(npz_name)
            pc = np.load(os.path.join(npz_folder, npz_name), allow_pickle=True)["pointcloud"][:, :3] @ rot_y
            img = cv2.imread(os.path.join(lbl_folder, sorted(os.listdir(lbl_folder))[i]), 0)
            plt.imshow(img)
            #plt.imshow(img > 20)
            plt.show()
            label = get_tree_indices(pc, img)

            #import pdb
            #pdb.set_trace()

            idx = np.random.permutation(len(pc))[:1024]
            (pc, label) = map(lambda x : x[idx], (pc, label))
            show_semantic(label, pc, view='top')
            plt.show()

def GenerateTopView(rot_y, viz=False):
    root = "."
    src_folder_names = ['train', 'val', 'test']

    for folder_name in src_folder_names:
        print(folder_name)
        npz_folder = os.path.join(root, folder_name, "npz")
        out_folder = os.path.join(root, folder_name, "pc_label")
        for i, npz_name in enumerate(sorted(os.listdir(npz_folder))):
            npz_path = os.path.join(npz_folder, npz_name)
            pc = np.load(npz_path, allow_pickle=True)["pointcloud"][:, :3] @ rot_y
            
            print("showing {}-th npz = {}".format(i, npz_name))
            
            top = topview(pc)
            out_path = os.path.join(out_folder, npz_name[:-len(".npz")]) + ".png"
            cv2.imwrite(out_path , top)

            if viz is True:            
                plt.figure(1)
                plt.switch_backend('TkAgg') #TkAgg (instead Qt4Agg)
                for view in ['top']:
                    idx = np.random.permutation(len(pc))[:1024]
                    pc  = pc[idx]
                    show_semantic(np.zeros(len(pc)), pc, view=view)
                    mng = plt.get_current_fig_manager()
                    mng.resize(*mng.window.maxsize())
                    plt.pause(0.5)
            
if __name__ == '__main__':
    

    degree = -4
    angle  = (degree/180)*pi
    rot_y = np.array([  [ cos(angle), 0, sin(angle)],
                        [          0, 1,         0] ,
                        [-sin(angle), 0, cos(angle)]])

    GenerateTopView(rot_y)
    #EvaluateLabel(rot_y)


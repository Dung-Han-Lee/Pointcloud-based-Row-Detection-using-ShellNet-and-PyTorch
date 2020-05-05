from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
import numpy as np
from math import pi, radians, degrees, cos, sin
import sys
import cv2
import os

DEMO_IMAGE_WIDTH = 500
DEMO_IMAGE_HEIGHT = 500
VELODYNE_NUM_BEAMS = 16
VELODYNE_VERTICAL_MAX_ANGLE = radians(15)

#TODO
# use max z height to label trees


def deproject_points(label, rimg, theta_step=0.35):
    """
    Args:
        label:(H x W) numpy semantic annotation of target 
        rimg: (H x W) numpy range image with intensity corresponds to range 
        theta step: resolution in theta (horizontal)
    Returns:
        pointcloud (N X 3) corresponding to label pixels (x, y, z)
    """

    # Segment ROI
    rimg[ label == 0 ] = 0
    
    h, w = rimg.shape
    compressed = np.zeros((int(h/4), w))
    for i in range(h):
        if(i%4==0):
            compressed[int(i/4)] = rimg[i]
    rimg = compressed
    h, w = compressed.shape

    # Map to (x,y,z) with r, theta, phi
    out = []
    for row in range(h):    # [0, 16] --> [15, -15]
        phi = VELODYNE_VERTICAL_MAX_ANGLE - radians(2 * row) 
        for col in range(w):    #[0, 511] --> [-pi/2, pi/2]
            theta = radians((w/2 - col)*theta_step)
            r = rimg[row][col]
            if r > 0: 
                z = r * sin(phi)
                r = r * cos(phi)
                x = r * cos(theta)
                y = r * sin(theta)
                out.append([x, y, z])

    return np.vstack(out)


def get_point_labels(label, rimg):
    """
    Args:
        label:(H x W) numpy semantic annotation of target 
        rimg: (H x W) numpy range image with intensity corresponds to range 
        theta step: resolution in theta (horizontal)
    Returns:
        pointcloud (N X 3) corresponding to label pixels (x, y, z)
    """

    h, w = rimg.shape
    compressed_rimg = np.zeros((int(h/4), w))
    compressed_lbl = np.zeros((int(h/4), w))
    for i in range(h):
        if(i%4==0):
            compressed_rimg[int(i/4)] = rimg[i]
            compressed_lbl[int(i/4)] = label[i]
    rimg = compressed_rimg
    label = compressed_lbl
    h, w = compressed_rimg.shape

    # Map to (x,y,z) with r, theta, phi
    out = []
    point_label = []
    for row in range(h):
        for col in range(w):   
            r = rimg[row][col]
            if r > 0:
                v = 0 if label[row][col] == 0 else 1
                point_label.append(v)

    return np.array(point_label)

def show_semantic(label, pointcloud, view='front'):
    """
    Args:
        label      : (N, ) numpy semantic annotation of target of two classes [0, 1]
        pointcloud : (N,4) numpy pointclouds [x, y, z]
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Draw class 0
    pc = pointcloud[label==0]
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c='b', marker='o', alpha=0.5)

    # Draw class 1
    pc = pointcloud[label==1]
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=3, c='r', marker='o')

    for xb, yb, zb in zip(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2]):
       ax.plot([xb], [yb], [zb], 'w')

    if view == 'front':
        ax.view_init(0, -180)
    elif view == 'top':
        ax.view_init(-90, 0)
    elif view == 'side':
        ax.view_init(0, 90)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim3d(0,40)
    ax.set_ylim3d(-20,20)
    ax.set_zlim3d(-3,3)
    plt.show()

def topview(pointcloud, scale=10, bin=True):
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
    row_range = DEMO_IMAGE_HEIGHT
    col_range = DEMO_IMAGE_WIDTH

    img = np.zeros((row_range,col_range)).astype(np.uint8)
    (x, y, z) = (pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2])

    # Scale values (c=0 corres to center line)
    c = scale*y + col_range/2
    r = x*scale
    z = 255 * ((z - min(z))/(max(z) - min(z)))

    # Mask out invalid points
    valid = (r >= 0) & (r < row_range-1) &\
            (c >= 0) &  (c < col_range-1)
    r = np.rint(r).astype('int')
    c = np.rint(c).astype('int')
    z = np.rint(z).astype('int')
    rvalid = r[valid]
    yvalid = c[valid]
    zvalid = 1 if bin is True else z[valid]
    img[rvalid, yvalid] = zvalid

    # Flip the image so r=0 is at btm, c=0 at right
    img = np.fliplr(img)
    img = np.flipud(img)

    return img


if __name__ == '__main__':
    root = "."
    label_dir = os.path.join(root, 'label')
    range_dir = os.path.join(root, 'range')
    fname_labels = sorted(os.listdir(label_dir))
    fname_rimgs  = sorted(os.listdir(range_dir))

    assert len(fname_labels) == len(fname_rimgs),\
        "each label must have a corresponding label, and vice versa"

    angle = -pi/60.
    rot_y = np.array([  [cos(angle), 0, sin(angle)],
                        [         0, 1,         0],
                        [-sin(angle), 0, cos(angle)]])


    for i in range(len(fname_labels)):
        path_label = os.path.join(label_dir, fname_labels[i])
        path_rimg  = os.path.join(range_dir, fname_rimgs[i])
        label_img = cv2.imread(path_label, 0)
        rimg  = np.load(path_rimg, allow_pickle=True)

        pc = deproject_points(1, rimg)
        #pc = pc @ rot_y
        
        label_pc = get_point_labels(label_img, rimg)
        idx = np.random.permutation(len(pc))[:1024]
        (pc, label_pc) = map(lambda x : x[idx], (pc, label_pc))
        show_semantic(label_pc, pc, view='front')
        plt.imshow(topview(pc))
        plt.show
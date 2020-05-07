from sys import argv
import os

def rename(directory):
    num = 0
    for filename in sorted(os.listdir(directory)):
        ext = filename.split(".")[-1]
        if (ext != 'png' and ext != 'npy' and ext != 'jpg') :
            continue
        os.rename( os.path.join(directory, filename),\
                os.path.join(directory, ("%04d" % num ) + "." + ext) )
        num +=1

root = "." if len(argv) < 2 else argv[1]
folders = ["train", "val", "test"]
subfolders = ["pc_label", "pointcloud"]
for folder in folders:
    for subfolder in subfolders:
        dir_name = os.path.join(root, folder, subfolder)
        rename(dir_name)  
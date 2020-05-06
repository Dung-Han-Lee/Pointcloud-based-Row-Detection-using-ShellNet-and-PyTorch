import os
import shutil
from glob import glob
from sys import argv

npz_dir = argv[1] if len(argv) == 2 else "./npz"
ext = ".npz"
npz_paths = [y for x in os.walk(npz_dir) for y in glob(os.path.join(x[0], '*' + ext))]

root = "."
src_folder_names = ['train', 'val', 'test']
for folder_name in src_folder_names:
    label_path = os.path.join(root, folder_name, "label")
    out_path = os.path.join(root, folder_name, "npz")
    for label_name in os.listdir(label_path):
        target_name = label_name[:-len("_visualize.png")] + ext
        for npz_path in npz_paths:
            npz_name = npz_path[-len(target_name):]
            if target_name == npz_name:
                shutil.copy2(npz_path, os.path.join(out_path, npz_name))

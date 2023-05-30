import os
import csv

import numpy as np
from PIL import Image
from SaveROI import segmented_path
from Helper import datapath, getCsvPath, checkDirectory

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def readtrafficsigns(rootpath):
    images = []  # images
    labels = []  # corresponding labels
    # loop over all
    for c in range(0, 2):
        prefix = rootpath + '/' + str(c) + '/'  # subdirectory for class
        gtfile = open(getCsvPath(prefix)[0])  # annotations file
        gtreader = csv.reader(gtfile, delimiter=',')  # csv parser for annotations file
        gtreader.__next__()  # skip header
        print("Starting process on folder-{}".format(c))
        # loop over all images in current annotations file
        for row in gtreader:
            print(row[0], row[-1], c)
            image = Image.open(prefix + row[0])
            images.append(np.asarray(image, dtype=np.uint8))  # First column : Filename
            labels.append(row[-1])  # Last column : ClassId
        gtfile.close()
        print("Folder-{} finished".format(c))
    return images, labels


images_labels = "Images_Labels"
save_path = os.path.join(datapath, images_labels)
checkDirectory(save_path)

# Path & var
image_file = "images.npy"
label_file = "labels.npy"
images_path = os.path.join(save_path, image_file)
labels_path = os.path.join(save_path, label_file)

if os.path.isfile(images_path) & os.path.isfile(labels_path):
    print("Opening {} & {} ...".format(image_file, label_file))
    images_np = np.load(images_path, allow_pickle=True)
    print("Images : {}".format(len(images_np)))
    labels_np = np.load(labels_path, allow_pickle=True, mmap_mode='r')
    print("Labels : {}".format(len(labels_np)))

else:
    print("[INFO] {} & {} not found".format(image_file, label_file))
    print("[INFO] training images and labels are read from the dataset directory")
    images_np, labels_np = readtrafficsigns(segmented_path)
    print("[INFO] Images save to {} for further use".format(images_path))
    np.save(images_path, images_np)
    print("[INFO] Labels save to {} for further use".format(labels_path))
    np.save(labels_path, labels_np)
    print("Finished.")

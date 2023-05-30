# import os
import sys
import math
import numpy as np
# from Loadimages import *
from Helper import *
from FeatureExtraction import *
from timeit import default_timer as timer
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


hog_dir = "HOGFeatures"
hog_feature = "HOG_feat.npy"
hog_visual = "HOG_vis.npy"

# Creating folder for feature and visual
feature_path = os.path.join(datapath, hog_dir)
checkDirectory(feature_path)

# HOG Features and Visualization
hog_feature_path = os.path.join(feature_path, hog_feature)
hog_visual_path = os.path.join(feature_path, hog_visual)

if os.path.isfile(hog_feature_path) & os.path.isfile(hog_visual_path):
    print("[INFO] Load Haar-HOG file")
    haarhogfeat = np.load(hog_feature_path, allow_pickle=True, mmap_mode='r')
    haarhogvisual = np.load(hog_visual_path, allow_pickle=True, mmap_mode='r')
    print("HoG features dimensions: {}".format(haarhogfeat.shape))
    print("HoG visual dimensions: {}".format(haarhogvisual.shape))
else:
    from ExtractHaar import haar_feat
    print("\n[INFO] {} & {} not found".format(hog_feature, hog_visual))
    feature = []
    visual = []
    tempft = []  # temporal array for every Haar Sub-signal (feature)
    tempvis = []  # temporal array for every Haar Sub-signal (visual)
    start = timer()
    print("\nExtracting HOG from Haar")
    for i in range(0, len(haar_feat)):
        for coeff in haar_feat[i]:
            tempft.append(hogextract(coeff)[0])  # LL, LH, HL, HH
            tempvis.append(hogextract(coeff)[1])  # LL, LH, HL, HH
        feature.append(tempft)
        visual.append(tempvis)
        tempft = []
        tempvis = []

        # Counter
        counter = (i + 1) / len(haar_feat)
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * counter), 100 * counter))
        sys.stdout.write(" |Image-{}|".format(i))
        sys.stdout.flush()

    print("\nExtracting HOG finished")
    print("\n[INFO] Saving {} ...".format(hog_feature))
    np.save(hog_feature_path, feature)
    print("[INFO] {} saved".format(hog_feature))
    feature = []
    print("\n[INFO] Saving {} ...".format(hog_visual))
    np.save(hog_visual_path, visual)
    print("[INFO] {} saved".format(hog_visual))
    visual = []

    end = timer()
    print("\nProcesses Completed!")
    ttime = end - start  # total time
    hours = math.floor(ttime // 3600)
    minutes = math.floor((ttime // 60) % 60)
    seconds = ttime % 60
    print("Time elapsed = {}:{}:{}".format(hours, minutes, seconds))

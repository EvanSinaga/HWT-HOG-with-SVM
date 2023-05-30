import sys
import math
import numpy as np

# from PIL import Image
from Helper import *
from FeatureExtraction import *
from timeit import default_timer as timer
from Undersample import X_resampled, y_resampled


haar_dir = "HaarFeatures"
haar_file = "HaarFeatures.npy"

feature_path = os.path.join(datapath, haar_dir)
print(feature_path)
checkDirectory(feature_path)

# Haar Wavelet Transform
hwt_path = os.path.join(feature_path, haar_file)

if os.path.isfile(hwt_path):
    print("[INFO] File {} exist".format(haar_file))
    print("[INFO] Loading from file ... ")
    haar_feat = np.load(hwt_path, allow_pickle=True, mmap_mode='r')
    print("HWT features loaded from {} to variable ==> haar_feat".format(haar_file))
else:
    print("[INFO] {} not found".format(haar_file))
    haar_feat = []
    start = timer()
    for i, image in enumerate(X_resampled):
        image = np.array(image[0])
        processed = preprocessing(image)
        hwt = haar(processed)
        haar_feat.append(hwt)
        # save the features using numpy save with .npy extention
        # which reduced the storage space by 4times compared to pickle
        # Progress Bar
        counter = (i + 1) / len(X_resampled)
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * counter), 100 * counter))
        sys.stdout.write(" |{}|".format(y_resampled[i]))
        sys.stdout.flush()
        # sleep(0.25)
    print("\nHaar Extraction Process Completed!")
    print("[INFO] Saving {} ...".format(haar_file))
    np.save(hwt_path, haar_feat)
    print("[INFO] {} saved".format(haar_file))

    end = timer()
    ttime = end - start  # total time
    hours = math.floor(ttime // 3600)
    minutes = math.floor((ttime // 60) % 60)
    seconds = ttime % 60
    print("Time elapsed = {}:{}:{}".format(hours, minutes, seconds))

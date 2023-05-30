import os
import joblib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from Helper import datapath
from Undersample import y_resampled
from sklearnex import patch_sklearn; patch_sklearn()
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Checking or creating directory for Classifier
clf_dir = "Classifier"
# Creating folder for feature and visual
clf_path = os.path.join(datapath, clf_dir)

features = np.load("E:/Skripsi_Fix/HOGFeatures/Concat.npy", allow_pickle=True, mmap_mode='r')
# features = np.load("E:/Skripsi_Fix/HOGFeatures/Concat_ND.npy", allow_pickle=True, mmap_mode='r')
# features = np.load("E:/Skripsi_Fix/HOGFeatures/Concat_NLL.npy", allow_pickle=True, mmap_mode='r')
# features = np.load("E:/Skripsi_Fix/HOGFeatures/Concat_LL.npy", allow_pickle=True, mmap_mode='r')
labels = y_resampled

# Split train and test data to 80% train and 20% test
(trainData, testData, trainLabels, testLabels) = \
    train_test_split(features, labels, test_size=0.2, random_state=42)

# show the sizes of each data split
print("Training data points: {}".format(len(trainLabels)))
print("Testing data points: {}".format(len(testLabels)))

# HOG-SVM file path
svc_pkl = "HOG_SVC_RBF.pkl"
clf_file = os.path.join(clf_path, svc_pkl)

print("[INFO] Loading classifier: SVC trained on HoG features...")
svc = joblib.load(clf_file, mmap_mode='r')
print("[INFO] Classifer is loaded as instance ::svc::")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
data_frame = pd.DataFrame(svc.cv_results_)

print("Best Estimator: \n{}\n".format(svc.best_estimator_))
print("Best Parameters: \n{}\n".format(svc.best_params_))
print("Best Test Score: \n{}\n".format(svc.best_score_))

predictions = svc.predict(testData)
print(classification_report(testLabels, predictions, digits=4))
c_matrix = confusion_matrix(testLabels, predictions)
print("Confusion Matrix: \n {}".format(c_matrix))

print("debug")


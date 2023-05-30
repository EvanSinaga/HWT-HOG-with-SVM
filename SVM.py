import os
import math
import joblib
import numpy as np
import pandas as pd

# Classification packages
from sklearn.svm import LinearSVC, SVC
from sklearnex import patch_sklearn; patch_sklearn()
from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.pipeline import make_pipeline
from timeit import default_timer as timer
import matplotlib.pyplot as plt

# Local functions / variables
from Helper import datapath, checkDirectory
from Undersample import y_resampled

# Checking or creating directory for Classifier
clf_dir = "Classifier"
# Creating folder for feature and visual
clf_path = os.path.join(datapath, clf_dir)
checkDirectory(clf_path)

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
svc_pkl = "HOG_SVC_X.pkl"
clf_file = os.path.join(clf_path, svc_pkl)

if os.path.isfile(clf_file):
    print("[INFO] Loading classifier: SVC trained on HoG features...")
    svc = joblib.load(clf_file, mmap_mode='r')
    print("[INFO] Classifer is loaded as instance ::svc::")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    data_frame = pd.DataFrame(svc.cv_results_)
    print("CV keys: \n{}\n".format(svc.cv_results_.keys()))
    print("CV Results: \n{}\n".format(data_frame))

    print("Best Estimator: \n{}\n".format(svc.best_estimator_))
    print("Best Parameters: \n{}\n".format(svc.best_params_))
    print("Best Test Score: \n{}\n".format(svc.best_score_))
    print("Support Vector: \n{}\n".format(svc.n_splits_))

    print("Accuracy on test data: {}\n".format(svc.score(testData, testLabels)))
    predictions = svc.predict(testData)
    print(classification_report(testLabels, predictions, digits=4))
    c_matrix = confusion_matrix(testLabels, predictions)
    print("Confusion Matrix: \n {}".format(c_matrix))

else:
    start = timer()
    print("\n[INFO] Pre-trained classifier not found. \n Training Classifier SVC")
    scaler = StandardScaler()
    svc = SVC(class_weight='balanced', max_iter=10000,
              cache_size=2000, random_state=42, verbose=1)
    print("Parameters : {}".format(svc.get_params().keys()))
    model = make_pipeline(scaler, svc, verbose=True)
    tuned_parameters = [{
                            'svc__C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
                            'svc__kernel': ['linear']
                        },
                        # {
                        #     'svc__C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
                        #     'svc__gamma': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
                        #     'svc__kernel': ['rbf']
                        # }
                        ]
    scorer = make_scorer(accuracy_score)
    grid = GridSearchCV(estimator=model, param_grid=tuned_parameters, scoring=scorer,
                        refit=True, verbose=10, error_score='raise')
    grid.fit(trainData, trainLabels)

    print("Saving grid model ...")
    joblib.dump(grid, clf_file)
    print("File Saved.")

    print("Best Estimator: \n{}\n".format(grid.best_estimator_))
    print("Best Parameters: \n{}\n".format(grid.best_params_))
    print("Best Test Score: \n{}\n".format(grid.best_score_))

    print("Accuracy on train data: {}".format(grid.score(trainData, trainLabels)))
    print("Accuracy on test data: {}\n".format(grid.score(testData, testLabels)))

    predictions = grid.predict(testData)

    c_report = classification_report(testLabels, predictions, digits=3)
    print("Classification Report: \n {}".format(c_report))
    c_matrix = confusion_matrix(testLabels, predictions)
    print("Confusion Matrix: \n {}".format(c_matrix))

    end = timer()
    print("\nProcesses Completed!")
    ttime = end - start  # total time
    hours = math.floor(ttime // 3600)
    minutes = math.floor((ttime // 60) % 60)
    seconds = ttime % 60
    print("Time elapsed = {}:{}:{}".format(hours, minutes, seconds))

print("Debug")

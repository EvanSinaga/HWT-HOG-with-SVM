from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from LoadImages import images_np, labels_np
# from sklearnex import patch_sklearn; patch_sklearn()
# from sklearn.model_selection import train_test_split


# Undersampling negative set
counter = Counter(labels_np)
print(counter)
X = images_np.reshape(-1, 1)
y = labels_np
rus = RandomUnderSampler(sampling_strategy='majority', random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
print("After undersampling: ", sorted(Counter(y_resampled).items()))

# # Split dataset into 80% train and 20% test set
# print("Splitting training and testing")
# trainData, testData, trainLabels, testLabels = \
#     train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
#
# # show the sizes of each data split
# print("Training data points: {}".format(len(trainLabels)))
# print("Testing data points: {}".format(len(testLabels)))

print("Debug")

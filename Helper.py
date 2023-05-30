import os
import cv2


# List of folder path
datapath = "E:/Skripsi_Fix/"
gtsdb = "E:/Skripsi-Dataset/PNG-GTSDB/"
segmented_path = "E:/Skripsi-Dataset/Segmented/"
imglist = [entry for entry in os.scandir(gtsdb)]


# Convert color space
def convertColor(image, argument):
    colorspace = {
        "rgb": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        "hsv": cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
        "gray": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    }
    return colorspace.get(argument, "False Argument")


# Check directory if it already exist
# Create directory if it is not
def checkDirectory(dir_path):
    if os.path.exists(dir_path):
        print("Directory {} Exist!".format(dir_path))
    else:
        os.mkdir(dir_path)
        print("Folder {} Created!\n".format(dir_path))


# Get csv path from folder
def getCsvPath(path):
    return [os.path.join(path, f).replace("\\", "/") for f in os.listdir(path) if f.endswith('.csv')]


# Bilateral Filter
def smoothing(image):
    filtered = cv2.bilateralFilter(src=image, d=9, sigmaColor=75, sigmaSpace=75)
    return filtered


def enlarge(image, scale):
    width = int(image.shape[1] * scale / 100)
    height = int(image.shape[0] * scale / 100)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

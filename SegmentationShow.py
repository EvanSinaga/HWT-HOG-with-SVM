import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Helper import smoothing

street_path = "E:/Skripsi-Dataset/PNG-GTSDB"
street_img_list = os.listdir(street_path)

street = cv2.imread(street_path + '/' + street_img_list[1])

street_rgb = cv2.cvtColor(street, cv2.COLOR_BGR2RGB)
street_hsv = cv2.cvtColor(street, cv2.COLOR_BGR2HSV)
grayscaleImg = cv2.cvtColor(street, cv2.COLOR_BGR2GRAY)

bilateral = smoothing(street_rgb)
plt.subplot(1, 2, 1)
plt.imshow(street_rgb)
plt.subplot(1, 2, 2)
plt.imshow(bilateral)
plt.show()





fig = plt.figure(figsize=(10, 7))
fig.suptitle("Color transformation")
fig.add_subplot(2, 2, 1)
plt.imshow(street_rgb)
plt.title("RGB")

fig.add_subplot(2, 2, 2)
plt.imshow(street_hsv)
plt.title("HSV")

fig.add_subplot(2, 2, 3)
plt.imshow(grayscaleImg, cmap="gray")
plt.title("Grayscale")
plt.show()

# OpenCv HSV only has 8-bit, so to fit in Hue Value, one must halves it
# Picking out a Range
# Blue | 90 >= H >= 130
lower_blue = np.array([90, 110, 30])
upper_blue = np.array([150, 255, 255])
# Red | H <= 10 or H >= 170
lower_red = np.array([173, 90, 20])
upper_red = np.array([180, 255, 255])
lower_red2 = np.array([0, 90, 20])
upper_red2 = np.array([7, 255, 255])
# Yellow : 10 >= H >= 30
lower_yellow = np.array([10, 120, 0])
upper_yellow = np.array([30, 255, 255])

# Masking + Result
blue_mask = cv2.inRange(street_hsv, lower_blue, upper_blue)
red_mask = cv2.inRange(street_hsv, lower_red, upper_red)
red_mask2 = cv2.inRange(street_hsv, lower_red2, upper_red2)
yellow_mask = cv2.inRange(street_hsv, lower_yellow, upper_yellow)

# Combine Mask
red_total = red_mask + red_mask2
blue_red = blue_mask + red_total

red_result = cv2.bitwise_and(street_rgb, street_rgb, mask=red_total)
blue_result = cv2.bitwise_and(street_rgb, street_rgb, mask=blue_mask)
yellow_result = cv2.bitwise_and(street_rgb, street_rgb, mask=yellow_mask)

bluered_result = cv2.bitwise_and(street_rgb, street_rgb, mask=blue_red)
final_result = cv2.bitwise_or(bluered_result, yellow_result)
# Change to grayscale
red2gray = cv2.cvtColor(red_result, cv2.COLOR_RGB2GRAY)
blue2gray = cv2.cvtColor(blue_result, cv2.COLOR_RGB2GRAY)
yellow2gray = cv2.cvtColor(yellow_result, cv2.COLOR_RGB2GRAY)
rgb2gray = cv2.cvtColor(final_result, cv2.COLOR_RGB2GRAY)

plt.suptitle("Final Mask")
plt.subplot(1, 2, 1)
plt.imshow(final_result)
plt.title("RBY Mask")
plt.subplot(1, 2, 2)
plt.imshow(rgb2gray, cmap="gray")
plt.title("Grayscale")
plt.show()

# Plot
# Blue Plot
bfig = plt.figure(figsize=(10, 10))
bfig.suptitle("Blue Filter")
bfig.add_subplot(2, 2, 1)
plt.imshow(street_rgb)
plt.title("Source Image")
bfig.add_subplot(2, 2, 2)
plt.imshow(blue_mask)
plt.title("Blue Mask")
bfig.add_subplot(2, 2, 3)
plt.imshow(blue_result)
plt.title("Blue Result")
bfig.add_subplot(2, 2, 4)
plt.imshow(blue2gray, cmap="gray")
plt.title("Gray Cmap")
# Red Plot
rfig = plt.figure(figsize=(10, 10))
rfig.suptitle("Red Filter")
rfig.add_subplot(2, 2, 1)
plt.imshow(street_rgb)
plt.title("Source Image")
rfig.add_subplot(2, 2, 2)
plt.imshow(red_mask)
plt.title("Red Mask")
rfig.add_subplot(2, 2, 3)
plt.imshow(red_result)
plt.title("Red Result")
rfig.add_subplot(2, 2, 4)
plt.imshow(red2gray, cmap="gray")
plt.title("Gray Cmap")
# Yellow Plot
yfig = plt.figure(figsize=(10, 10))
yfig.suptitle("Yellow Filter")
yfig.add_subplot(2, 2, 1)
plt.imshow(street_rgb)
plt.title("Source Image")
yfig.add_subplot(2, 2, 2)
plt.imshow(yellow_mask)
plt.title("Yellow Mask")
yfig.add_subplot(2, 2, 3)
plt.imshow(yellow_result)
plt.title("Yellow Result")
yfig.add_subplot(2, 2, 4)
plt.imshow(yellow2gray, cmap="gray")
plt.title("Gray Cmap")
plt.show()


# Erotion + Dilation
# Construct a rectangular kernel from the current size and then
# kernelSizes = [(3, 3)]

# Apply an "opening" operation (Erotion + Dilation)
def opening(image, kernelSize=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    # Opening
    opening_result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # Dilation
    opening_result = cv2.dilate(opening_result, kernel)
    # Change to binary
    (thresh, img_binary) = cv2.threshold(opening_result, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img_binary


morph_results = opening(rgb2gray)
# loop over the kernels sizes
rows = 2
column = 2
# mfig = Morphology Figure
mfig = plt.figure(constrained_layout=True)
mfig.suptitle("Morfologi")

mfig.add_subplot(rows, column, 1)
plt.imshow(street_rgb)
plt.title("RGB")

mfig.add_subplot(rows, column, 2)
plt.imshow(street_hsv)
plt.title("HSV")

mfig.add_subplot(rows, column, 3)
plt.imshow(morph_results)
plt.title("Morph Result")

mfig.add_subplot(rows, column, 4)
plt.imshow(morph_results, cmap="gray")
plt.title("Binary")

plt.show()


def cannyimg(image, sigma=0.33):
    # Applied Bilateral Filter
    image = cv2.bilateralFilter(src=image, d=9, sigmaColor=75, sigmaSpace=75)
    # Generate auto canny
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


# Canny edge
canny_edge = cannyimg(morph_results)
plt.suptitle("Canny edge")
plt.subplot(1, 1, 1)
plt.imshow(canny_edge, cmap='gray')
plt.show()


# Bounding Box
def bbox(rgbimage, cannyedge):
    bbox = rgbimage.copy()
    mask = np.zeros(rgbimage.shape[:2], dtype=rgbimage.dtype)
    (contours, _) = cv2.findContours(cannyedge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = contours[0] if len(contours) == 2 else contours[1]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / h
        if area >= (25 * 25) and 0.5 <= ratio <= 1.5:
            # if area >= (50 * 50) and 0.75 <= ratio <= 1.25:
            cv2.drawContours(mask, [cnt], -1, (255, 0, 0), -1)
            cv2.rectangle(bbox, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi_image = cv2.bitwise_or(rgbimage, rgbimage, mask=mask)
    return bbox, roi_image

bounding_box = bbox(street_rgb, canny_edge)
box = bounding_box[0]
roi = bounding_box[1]

# Show Bounding Box
bbfig = plt.figure(constrained_layout=True)
bbfig.suptitle("ROI Result")
plt.subplot(1, 2, 1)
plt.imshow(box)
plt.title("Bounding Box")
plt.subplot(1, 2, 2)
plt.imshow(roi)
plt.title("ROI Image")
plt.show()

# # PRINTING COLOR SPACE CHANNEL
# print("R : ", street_rgb[200, 200, 0])
# print("G : ", street_rgb[200, 200, 1])
# print("B : ", street_rgb[200, 200, 2])
#
# print("H : ", street_hsv[200, 200, 0])
# print("S : ", street_hsv[200, 200, 1])
# print("V : ", street_hsv[200, 200, 2])
#
# print("R : ", street_rgb[:, :, 0])
# print("G : ", street_rgb[:, :, 1])
# print("B : ", street_rgb[:, :, 2])
#
# print("H : ", street_hsv[:, :, 0])
# print("S : ", street_hsv[:, :, 1])
# print("V : ", street_hsv[:, :, 2])

print("Debug")

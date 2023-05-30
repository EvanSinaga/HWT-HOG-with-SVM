import cv2
import numpy as np
from Helper import convertColor, smoothing


class ColorSegmentation:
    def __init__(self, rgb, hsv, kernelsize=(3, 3)):
        self.rgb = rgb
        self.hsv = hsv
        self.kernelsize = kernelsize

    def masking(self):
        # OpenCv HSV only has 8-bit, so to fit in Hue Value, one must halve it
        # Picking out a Range
        # Blue : 90 >= H >= 130
        lower_blue = np.array([90, 110, 20])
        upper_blue = np.array([130, 255, 255])
        # Red : H <= 10 or H >= 170
        lower_red = np.array([173, 90, 20])
        upper_red = np.array([180, 255, 255])
        lower_red2 = np.array([0, 90, 20])
        upper_red2 = np.array([7, 255, 255])
        # Yellow : 10 >= H >= 30
        lower_yellow = np.array([10, 120, 0])
        upper_yellow = np.array([30, 255, 255])

        # # Threshold from (Chen, 2013)
        # # Blue : 90 >= H >= 130
        # lower_blue = np.array([60, 90, 14])
        # upper_blue = np.array([87, 255, 255])
        # # Red : H <= 10 or H >= 170
        # lower_red = np.array([120, 90, 21])
        # upper_red = np.array([180, 255, 255])
        # lower_red2 = np.array([0, 90, 21])
        # upper_red2 = np.array([5, 255, 255])
        # # Yellow : 10 >= H >= 30
        # lower_yellow = np.array([9, 105, 47])
        # upper_yellow = np.array([22, 255, 255])

        # Generate mask
        blue_mask = cv2.inRange(self.hsv, lower_blue, upper_blue)
        red_mask = cv2.inRange(self.hsv, lower_red, upper_red)
        red_mask2 = cv2.inRange(self.hsv, lower_red2, upper_red2)
        yellow_mask = cv2.inRange(self.hsv, lower_yellow, upper_yellow)

        # Masking + Result
        red_combine = red_mask + red_mask2
        blue_red_combine = blue_mask + red_combine
        blue_red_result = cv2.bitwise_and(self.rgb, self.rgb, mask=blue_red_combine)
        yellow_result = cv2.bitwise_and(self.rgb, self.rgb, mask=yellow_mask)
        final_result = cv2.bitwise_or(blue_red_result, yellow_result)
        return final_result

    # Erotion + Dilation
    # Apply an "opening" operation (Erotion + Dilation)
    def morphing(self):
        # Change to grayscale
        grayscale_img = convertColor(self.masking(), "gray")
        # Blur image to reduce noise
        blur = cv2.GaussianBlur(grayscale_img, (5, 5), 0)
        # Creating rectangle kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.kernelsize)
        # Applied opening
        opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
        # Dilation after Opening
        # open_dilat = cv2.dilate(opening, (2, 2))
        # Applied Thresholding
        (thresh, img_binary) = cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return img_binary


# Create region of interest on image
class ImageRoi:
    def __init__(self, binaryimg):
        self.binaryimg = binaryimg

    # Generate Canny image
    def cannyimg(self, sigma=0.33):
        # Applied Bilateral Filter
        image = smoothing(self.binaryimg)
        # Generate auto canny
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
        return edged

    def bbox(self, rgbimage):
        bbox = rgbimage.copy()
        mask = np.zeros(rgbimage.shape[:2], dtype=rgbimage.dtype)
        (contours, _) = cv2.findContours(self.cannyimg(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    def coordinates(self, rgbimage):
        (contours, _) = cv2.findContours(self.cannyimg(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi = [] # Image
        gt_roi = [] # Coordinates
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / h
            # Reposition
            ymin = max(0, y - 20)
            xmin = max(0, x - 20)
            ymax = y + h + 20
            xmax = x + w + 20
            if area >= (25 * 25) and 0.5 <= ratio <= 1.5:
                roi.append(rgbimage[ymin:ymax, xmin:xmax])
                gt_roi.append([w, h, xmin, ymin, xmax, ymax])
        return roi, gt_roi

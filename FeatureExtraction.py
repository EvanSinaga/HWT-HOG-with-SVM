import cv2
import pywt
from skimage.feature import hog


def preprocessing(image):
    # Convert image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize image to 256x256 pixel
    newimg = cv2.resize(grayscale, (256, 256), interpolation=cv2.INTER_LINEAR)
    return newimg


def haar(image):
    coeffs = pywt.dwt2(image, 'haar')
    ll, (hl, lh, hh) = coeffs
    return ll, hl, lh, hh


def hogextract(image):
    feature, hog_img = hog(image, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), visualize=True, multichannel=False)
    return feature, hog_img


# Inverse Haar
def invhaar(icoeffs):
    inverse = pywt.idwt2(icoeffs, 'haar')
    return inverse

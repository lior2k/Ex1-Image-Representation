"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import cv2
import numpy as np
import matplotlib.pyplot as plt

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int_:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 206284960


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = cv2.normalize(cv2.imread(filename), None, 0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
    if representation == LOAD_GRAY_SCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif representation == LOAD_RGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)
    representation = 'gray' if representation == 1 else None
    plt.imshow(img, cmap=representation)
    plt.show()


# Here is the RGB -> YIQ conversion:
#     [ Y ]     [ 0.299   0.587   0.114 ] [ R ]
#     [ I ]  =  [ 0.596  -0.275  -0.321 ] [ G ]
#     [ Q ]     [ 0.212  -0.523   0.311 ] [ B ]
def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    coeffs = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    return imgRGB @ coeffs.transpose()


# Here is the YIQ -> RGB conversion:
#     [ R ]     [ 1   0.956   0.621 ] [ Y ]
#     [ G ]  =  [ 1  -0.272  -0.647 ] [ I ]
#     [ B ]     [ 1  -1.105   1.702 ] [ Q ]
def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    coeffs = np.array([[1, 0.956, 0.621], [1, -0.272, -0.647], [1, -1.105, 1.702]])
    return imgYIQ @ coeffs.transpose()


def equalizeRGBImg(img: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    height, width, channels = img.shape
    num_pixels = height * width
    img = transformRGB2YIQ(img)
    img = np.around(img).astype('uint8')
    hist, bins = np.histogram(img[:, :, 0], range=(0, 255), bins=256)
    cumsum = np.cumsum(hist)
    lut = [int((cumsum[i] / num_pixels) * 255) for i in range(0, 256)]
    for i in range(height):
        for j in range(width):
            img[i, j, 0] = lut[img[i, j, 0]]
    histeq, _ = np.histogram(img[:, :, 0], range=(0, 255), bins=256)
    img = transformYIQ2RGB(img)
    img = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
    return img, hist, histeq


def equalizeGrayImg(img: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    height, width = img.shape
    num_pixels = height * width
    hist, bins = np.histogram(img, range=(0, 255), bins=256)
    cumsum = np.cumsum(hist)
    lut = [int((cumsum[i] / num_pixels) * 255) for i in range(0, 256)]
    for i in range(height):
        for j in range(width):
            img[i, j] = lut[img[i, j]]
    histeq, _ = np.histogram(img, range=(0, 255), bins=256)
    img = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
    return img, hist, histeq


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    img = cv2.normalize(imgOrig, None, 0, 255, cv2.NORM_MINMAX)
    img = np.around(img).astype('uint8')
    if len(img.shape) == 2:
        return equalizeGrayImg(img)
    return equalizeRGBImg(img)


def quantizeGrayImg(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    hist, bins = np.histogram(imOrig, range=(0, 255), bins=256)
    z = np.percentile(hist, np.linspace(0, 100, nQuant + 1))
    print(z)
    images, errors = [], []
    for i in range(nIter):
        q = np.array([np.mean(hist[(hist >= z[j]) & (hist <= z[j + 1])]) for j in range(nQuant)])
        imQuant = np.zeros(hist.shape)
        for j in range(nQuant):
            imQuant[(hist >= z[j]) & (hist <= z[j + 1])] = q[j]
        mse = ((hist - imQuant) ** 2).mean()
        for j in range(1, len(z) - 1):
            z[j] = (q[j - 1] + q[j]) / 2

        images.append(imQuant / 255.0)
        errors.append(mse)

    return images, errors


# def quantizeRGBImg(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
#     imOrig = transformRGB2YIQ(imOrig)
#     images, errors = [], []
#     for i in range(nIter):
#         z = np.percentile(imOrig[:, :, 0], np.linspace(0, 100, nQuant + 1))
#         q = np.array([np.mean(imOrig[:, :, 0][(imOrig[:, :, 0] >= z[j]) & (imOrig[:, :, 0] <= z[j + 1])]) for j in range(nQuant)])
#         imQuant = imOrig.copy()
#         for j in range(nQuant):
#             imQuant[:, :, 0][(imOrig[:, :, 0] >= z[j]) & (imOrig[:, :, 0] <= z[j + 1])] = q[j]
#         mse = ((imOrig - imQuant) ** 2).mean()
#         for j in range(1, len(z) - 1):
#             z[j] = (q[j - 1] + q[j]) / 2
#         imOrig = imQuant.copy()
#
#         images.append(transformYIQ2RGB(imQuant) / 255.0)
#         errors.append(mse)
#
#     return images, errors

def quantizeRGBImg(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    currentImage = transformRGB2YIQ(imOrig)[:, :, 0]
    images, errors = [], []
    for i in range(nIter):
        z = np.percentile(currentImage, np.linspace(0, 100, nQuant + 1))
        q = np.array([np.mean(currentImage[(currentImage >= z[j]) & (currentImage <= z[j + 1])]) for j in range(nQuant)])
        imQuant = np.zeros_like(currentImage)
        for j in range(nQuant):
            imQuant[(currentImage >= z[j]) & (currentImage <= z[j + 1])] = q[j]
        mse = ((currentImage - imQuant) ** 2).mean()
        for j in range(1, len(z) - 1):
            z[j] = (q[j - 1] + q[j]) / 2
        currentImage = imQuant.copy()
        imQuant = np.dstack((imQuant, imOrig[:, :, 1], imOrig[:, :, 2]))
        images.append(transformYIQ2RGB(imQuant) / 255.0)
        errors.append(mse)

    return images, errors


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """

    # Check if the image is grayscale or RGB
    imOrig = cv2.normalize(imOrig, None, 0, 255, cv2.NORM_MINMAX)
    if len(imOrig.shape) == 2:
        return quantizeGrayImg(imOrig, nQuant, nIter)
    elif len(imOrig.shape) == 3 and imOrig.shape[2] == 3:
        return quantizeRGBImg(imOrig, nQuant, nIter)
    else:
        raise ValueError("Image must be grayscale or RGB")





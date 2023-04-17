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
    # load image and normalize values to [0:1]
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


# RGB -> YIQ conversion:
# [ Y ]     [ 0.299   0.587   0.114 ] [ R ]
# [ I ]  =  [ 0.596  -0.275  -0.321 ] [ G ]
# [ Q ]     [ 0.212  -0.523   0.311 ] [ B ]
def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    coeffs = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    return imgRGB @ coeffs.transpose()


# YIQ -> RGB conversion:
# [ R ]     [ 1   0.956   0.621 ] [ Y ]
# [ G ]  =  [ 1  -0.272  -0.647 ] [ I ]
# [ B ]     [ 1  -1.105   1.702 ] [ Q ]
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
    # map values using lut
    for i in range(height):
        for j in range(width):
            img[i, j, 0] = lut[img[i, j, 0]]
    # compute histogram of new equalized image
    histeq, _ = np.histogram(img[:, :, 0], range=(0, 255), bins=256)
    # transform back to rgb and normalize back to [0:255]
    img = transformYIQ2RGB(img)
    img = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
    return img, hist, histeq


def equalizeGrayImg(img: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    height, width = img.shape
    num_pixels = height * width
    hist, bins = np.histogram(img, range=(0, 255), bins=256)
    cumsum = np.cumsum(hist)
    lut = [int((cumsum[i] / num_pixels) * 255) for i in range(0, 256)]
    # map values using lut
    for i in range(height):
        for j in range(width):
            img[i, j] = lut[img[i, j]]
    histeq, _ = np.histogram(img, range=(0, 255), bins=256)
    # normalize back to [0:255]
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
    # grayscale image
    if len(img.shape) == 2:
        return equalizeGrayImg(img)
    # rgb image
    return equalizeRGBImg(img)


# compute the mean for each cell and return a list
def compute_quantization_values(hist: np.ndarray, nQuant: int, z: np.ndarray):
    q = []
    for j in range(nQuant):
        hist_range = hist[z[j]: z[j + 1]]
        w_avg = np.average(range(len(hist_range)), weights=hist_range)
        q.append(w_avg + z[j])
    return q


# compute boundaries such that each cell would contain roughly the same number of pixels
def compute_quantization_boundaries(hist, nQuant):
    cumhist = np.cumsum(hist)
    total_pixels = cumhist[-1]
    target_pixels_per_cell = total_pixels // nQuant

    boundaries = [0]
    num_cells = 1
    for i in range(256):
        if cumhist[i] >= target_pixels_per_cell * num_cells:
            boundaries.append(i)
            num_cells += 1
            if num_cells > nQuant:
                break

    return np.array(boundaries)


def quantizeChannel(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    hist, bins = np.histogram(imOrig, range=(0, 255), bins=256)
    z = compute_quantization_boundaries(hist, nQuant)

    images, errors = [], []
    for i in range(nIter):
        q = compute_quantization_values(hist, nQuant, z)

        # generate new quantized image
        imQuant = np.zeros_like(imOrig)
        for j in range(len(q)):
            imQuant[imOrig > z[j]] = q[j]
        images.append(imQuant / 255.0)

        # compute mse
        mse = ((imOrig - imQuant) ** 2).mean()
        errors.append(mse)

        # update z values (boundaries)
        for j in range(1, len(z) - 1):
            z[j] = (q[j - 1] + q[j]) / 2

    return images, errors


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """

    # normalize image values to 0-256
    imOrig = cv2.normalize(imOrig, None, 0, 255, cv2.NORM_MINMAX)

    # grayscale image
    if len(imOrig.shape) == 2:
        return quantizeChannel(imOrig, nQuant, nIter)

    # rgb image
    elif len(imOrig.shape) == 3 and imOrig.shape[2] == 3:
        yiq_image = transformRGB2YIQ(imOrig)
        yiq_images, error = quantizeChannel(yiq_image[:, :, 0], nQuant, nIter)
        rgb_images = [transformYIQ2RGB(np.dstack((img, yiq_image[:, :, 1], yiq_image[:, :, 2]))) for img in yiq_images]
        return rgb_images, error

    raise ValueError("Image must be grayscale or RGB")





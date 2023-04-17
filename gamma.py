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
from ex1_utils import LOAD_GRAY_SCALE

import cv2
import numpy as np


def gammaDisplay(img_path: str, rep: int) -> None:
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """

    def adjust_gamma(image, gamma):
        # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def on_trackbar(val):
        # convert the trackbar value to a gamma value between 0.01 and 2
        gamma = val / 100.0 + 0.01
        # apply gamma correction to the image
        gamma_corrected = adjust_gamma(img, gamma)
        # display the corrected image
        cv2.imshow("Gamma Correction", gamma_corrected)

    # load the image and convert it to the specified representation
    img = cv2.imread(img_path)
    if rep == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # create a window to display the image and the trackbar
    cv2.namedWindow("Gamma Correction", cv2.WINDOW_NORMAL)

    # create a trackbar with values from 0 to 200 (to account for the 0.01 resolution)
    cv2.createTrackbar("Gamma", "Gamma Correction", 100, 200, on_trackbar)

    # display the original image
    cv2.imshow("Gamma Correction", img)

    # wait for a key press and then clean up the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()

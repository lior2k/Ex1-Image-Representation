import numpy as np
# Here is the RGB -> YIQ conversion:
#
#     [ Y ]     [ 0.299   0.587   0.114 ] [ R ]
#     [ I ]  =  [ 0.596  -0.275  -0.321 ] [ G ]
#     [ Q ]     [ 0.212  -0.523   0.311 ] [ B ]
if __name__ == '__main__':
    rgb = np.array([0.3803922, 0.4784314, 0.5647059])
    coeffs = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    yiq = coeffs @ rgb
    print(yiq)

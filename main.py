import cv2 as cv
from fundamental_matrix import get_fundamental_matrix
from epipolar_lines import get_epipolar_line


img1 = cv.imread('right_img.jpg', cv.IMREAD_GRAYSCALE)  # right image
img2 = cv.imread('left_img.jpg', cv.IMREAD_GRAYSCALE)  # left image

F, mask = get_fundamental_matrix(img1, img2)
print(F)
line = get_epipolar_line((100, 100), F, range(img1.shape[1]), range(img1.shape[0]))

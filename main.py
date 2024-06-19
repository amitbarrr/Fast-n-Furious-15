import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from image_process import *
from sad_final import sad_test

WINDOW_SIZE = 9  # Size of the window to search for the best match in the SAD algorithm

DISTANCE_BETWEEN_CAMERAS = 0.2 # in meters
DISTANCE_FROM_CAMERA = 1 # of the object from the camera in meters
RULER_SIZE = 1 # in meters
ANGLE_OF_VIEW = np.arctan(RULER_SIZE / (2 * DISTANCE_FROM_CAMERA))

img1 = cv.imread('images/left_img.jpg')  # left image
img2 = cv.imread('images/right_img.jpg')  # right image

normalize_factor = (DISTANCE_BETWEEN_CAMERAS * img1.shape[1]) / (2 * np.tan(ANGLE_OF_VIEW / 2))
# print(normalize_factor)
draw_distance_map(img1, img2, normalize_factor, WINDOW_SIZE)

# Draw the epipolar lines on the right image and the original point on the left image
pixel = (200, 210)
line = get_parallel_line(pixel, range(30, 250))
# sad_test(img1, img2, pixel, line, 9)

# img3 = img1
# img4 = img2
# img1 = cv.circle(img1, (pixel[1], pixel[0]), 3, (255, 0, 0), -1)
# matching_pixels = sad(line, pixel, img1, img2, WINDOW_SIZE)
# print(matching_pixels)
# for i, matching_pixel in enumerate(matching_pixels):
#     img2 = cv.circle(img2, (matching_pixel[1], matching_pixel[0]), 3, (255, i * 45, i * 45), -1)
#
# match = matching_pixels[0]
# match_line = get_parallel_line(match, range(0, 250))
# img3 = cv.circle(img3, (pixel[1], pixel[0]), 3, (255, 0, 0), -1)
# matching_pixels = sad(line, pixel, img3, img4, WINDOW_SIZE)
# print(matching_pixels)
# for i, matching_pixel in enumerate(matching_pixels):
#     img2 = cv.circle(img2, (matching_pixel[1], matching_pixel[0]), 3, (255, i * 45, i * 45), -1)
#
# plt.subplot(121), plt.imshow(img1)
# plt.subplot(122), plt.imshow(img2)
# plt.show()

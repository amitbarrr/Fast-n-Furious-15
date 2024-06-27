import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from image_process import *
from sad_final import *

WINDOW_SIZE = 9  # Size of the window to search for the best match in the SAD algorithm

DISTANCE_BETWEEN_CAMERAS = 0.2 # in meters
DISTANCE_FROM_CAMERA = 1 # of the object from the camera in meters
RULER_SIZE = 1 # in meters
ANGLE_OF_VIEW = np.arctan(RULER_SIZE / (2 * DISTANCE_FROM_CAMERA))

img1 = cv.imread('images/left_img2.jpg')  # left image
img2 = cv.imread('images/right_img2.jpg')  # right image

img1 = img1[200:900, 300:1000]
img2 = img2[200:900, 300:1000]

normalize_factor = (DISTANCE_BETWEEN_CAMERAS * img1.shape[1]) / (2 * np.tan(ANGLE_OF_VIEW / 2))
# print(normalize_factor)
draw_distance_map(img1, img2, 9, normalize_factor=normalize_factor, cost="sad", alg="average", write=True)

# Draw the epipolar lines on the right image and the original point on the left image
# pixel = (100, 250)
# x_values = range(30, len(img1[0])-9)
# line = get_parallel_line(pixel, x_values)
# sad_test(img1, img2, pixel, line, 15)
# vals, points = sad(line, pixel, img1, img2, 15)
# print(points, points[:, 1])
# plt.plot(x_values, vals)
# marker_vals = [vals[p-30] for p in points[:, 1]]
# print(marker_vals, points[:, 1])
# plt.scatter(points[:, 1], marker_vals, color='red', zorder=5)
# plt.show()
# average_sad_test(img1, img2, pixel, 13, 31, 25, "ssd")

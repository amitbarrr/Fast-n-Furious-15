import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from image_process import *
from sad_final import sad_test


DISTANCE_BETWEEN_CAMERAS = 0.2
DISTANCE_FROM_CAMERA = 1
RULER_SIZE = 1
ANGLE_OF_VIEW = np.arctan(RULER_SIZE / (2 * DISTANCE_FROM_CAMERA))

img1 = cv.imread('images/left_img_in.png')  # left image
img2 = cv.imread('images/right_img_in.png')  # right image

normalize_factor = (DISTANCE_BETWEEN_CAMERAS * img1.shape[1]) / (2 * np.tan(ANGLE_OF_VIEW / 2))
print(normalize_factor)
draw_distance_map(img1, img2, normalize_factor, 31)

# Draw the epipolar lines on the right image and the original point on the left image
pt = (100, 150)
line = get_parallel_line(pt, range(50, img1.shape[1]))
# sad_test(img1, img2, pt, line, 11)

# img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
# img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
# img1 = cv.circle(img1, tuple(pt), 3, (255, 0, 0), -1)
# for point in line:
#     img2 = cv.circle(img2, (point[1], point[0]), 2, (0, 255, 0), -1)
#
# plt.subplot(121), plt.imshow(img1)
# plt.subplot(122), plt.imshow(img2)
# plt.show()

# def sad(point_list: list[tuple], point: tuple, image1: np.array, image2: np.array, size: int):
#     """
#
#     :param point_list: the line in which we want to find the point
#     :param point: the point we want to find
#     :param image1: the origin picture
#     :param image2: the picture in which we are looking
#     :param size: the size of neighborhood in which we are looking
#     :return: the position (tuple) of the best match
#     """
#     if point[0] > len(image1) or point[1] > len(image1[0]):
#         return None
#
#     padded_image1 = np.pad(image1, size, mode='constant', constant_values=0)
#     block1 = padded_image1[point[1]:point[1] + 2 * size + 1, point[0]: point[0] + 2 * size + 1]
#     padded_image2 = np.pad(image2, size, mode='constant', constant_values=0)
#
#     min_sad_value = np.inf
#     min_sad_pos = (0, 0)
#     for position in point_list:
#         x = position[1]
#         y = position[0]
#
#         block2 = padded_image2[y:y + 2 * size + 1, x:x + 2 * size + 1]
#         sad_val = np.sum(np.abs(block1 - block2))
#
#         if sad_val < min_sad_value:
#             min_sad_value = sad_val
#             min_sad_pos = (y, x)
#
#     return min_sad_pos

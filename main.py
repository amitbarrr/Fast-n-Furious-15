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
# print(normalize_factor)
draw_distance_map(img1, img2, normalize_factor, 9)

# Draw the epipolar lines on the right image and the original point on the left image
pt = (100, 250)
line = get_parallel_line(pt, range(0, 250))
# sad_test(img1, img2, pt, line, 9)

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
#     # Check for edge cases
#     if point[0] > len(image1) or point[1] > len(image1[0]):
#         return None
#     # Create
#     padded_image1 = np.pad(image1, size, mode='constant', constant_values=0)
#     block1 = padded_image1[point[0]:point[0] + 2 * size + 1, point[1]: point[1] + 2 * size + 1]
#     padded_image2 = np.pad(image2, size, mode='constant', constant_values=0)
#
#     min_sad_value = np.inf
#     sad_lst = np.zeros((5, 3))
#     # Run on all Points in the list
#     for position in point_list:
#         x, y = position[1], position[0]
#         # Create new neighborhood
#         block2 = padded_image2[y:y + 2 * size + 1, x:x + 2 * size + 1]
#         sad_val = np.sum(np.abs(block1 - block2))
#         if np.count_nonzero(sad_lst == 0) > 0:
#             sad_lst[-1] = np.array([y, x, sad_val])
#             sad_lst.sort(key=lambda a: a[2])
#             min_sad_value = sad_lst[-1][-1]
#         elif sad_val < min_sad_value:
#             sad_lst.append(np.array([y, x, sad_val]))
#             sad_lst.sort(key=lambda a: a[2])
#             min_sad_value = sad_lst[-1][-1]
#             sad_lst.pop(0)
#     sad_lst = np.array(sad_lst)[:, :-1]
#     return sad_lst
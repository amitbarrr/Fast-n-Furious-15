import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from image_process import *

DISTANCE_BETWEEN_CAMERAS = 1
DISTANCE_FROM_CAMERA = 1
RULER_SIZE = 1
ANGLE_OF_VIEW = np.arctan(RULER_SIZE / (2 * DISTANCE_FROM_CAMERA))

img1 = cv.imread('images/left_img_in.png', cv.IMREAD_GRAYSCALE)  # left image
img2 = cv.imread('images/right_img_in.png', cv.IMREAD_GRAYSCALE)  # right image

normalize_factor = (DISTANCE_BETWEEN_CAMERAS * img1.shape[1]) / (2 * np.tan(ANGLE_OF_VIEW / 2))
print(normalize_factor)

pt = (200, 200)

draw_distance_map(img1, img2, normalize_factor)

line = get_parallel_line(pt, range(50, img1.shape[1]))

# Draw the epipolar lines on the right image and the original point on the left image
img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
img1 = cv.circle(img1, tuple(pt), 3, (255, 0, 0), -1)
for point in line:
    img2 = cv.circle(img2, (point[1], point[0]), 2, (0, 255, 0), -1)

plt.subplot(121), plt.imshow(img1)
plt.subplot(122), plt.imshow(img2)
# plt.show()


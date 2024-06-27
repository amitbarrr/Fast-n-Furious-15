import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

from image_process import *
from sad_final import *

WINDOW_SIZE = 9  # Size of the window to search for the best match in the SAD algorithm

DISTANCE_BETWEEN_CAMERAS = 0.2 # in meters
DISTANCE_FROM_CAMERA = 1 # of the object from the camera in meters
RULER_SIZE = 1 # in meters
ANGLE_OF_VIEW = np.arctan(RULER_SIZE / (2 * DISTANCE_FROM_CAMERA))

IMAGE = 1

img1 = cv.imread(f'images/left_img{IMAGE}.jpg')  # left image
img2 = cv.imread(f'images/right_img{IMAGE}.jpg')  # right image

if IMAGE == 2:
    # Cutoff for img2
    img1 = img1[200:900, 300:1000]
    img2 = img2[200:900, 300:1000]
elif IMAGE == 3:
    # Cutoff for img3
    img1 = img1[200:500, 200:800]
    img2 = img2[200:500, 200:800]


# normalize_factor = (DISTANCE_BETWEEN_CAMERAS * img1.shape[1]) / (2 * np.tan(ANGLE_OF_VIEW / 2))
normalize_factor = 1
# print(normalize_factor)

NUM_WORKERS = 5

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    # List of future objects
    futures = [executor.submit(draw_distance_map, img1, img2, size, normalize_factor, "block", "ssd", True, True) for size in range(3, 15, 2)]

    # As tasks complete, print their result
    # for future in as_completed(futures):
    #     result = future.result()
    #     print(result)

# draw_distance_map(img1, img2, 3, normalize_factor=normalize_factor, cost="ssd", alg="block", write=True, disp=True)

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

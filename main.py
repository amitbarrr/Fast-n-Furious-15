import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from fundamental_matrix import get_fundamental_matrix, get_matches
from epipolar_lines import get_epipolar_line, get_parallel_line


img1 = cv.imread('left_img.jpg', cv.IMREAD_GRAYSCALE)  # left image
img2 = cv.imread('right_img.jpg', cv.IMREAD_GRAYSCALE)  # right image

F, mask = get_fundamental_matrix(img1, img2)
print(F)

pts1, pts2 = get_matches(img1, img2)
# We select only inlier points
# pts1 = pts1[mask.ravel()==1]
# pts2 = pts2[mask.ravel()==1]


pt = (200, 1069)

color = tuple(np.random.randint(0,255,3).tolist())

# line = get_epipolar_line(pt, F, range(img1.shape[1]))
line = get_parallel_line(pt, img1.shape[1])
print(line)
# Draw the epipolar lines on the right image and the original point on the left image
img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
img1 = cv.circle(img1, tuple(pt), 10, (255,0,0), -1)
for point in line:
    img2 = cv.circle(img2, (point[1], point[0]), 2, color, -1)

plt.subplot(121), plt.imshow(img1)
plt.subplot(122), plt.imshow(img2)
plt.show()

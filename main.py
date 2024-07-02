from image_process import *
from sad_final import *


DISTANCE_BETWEEN_CAMERAS = 0.2 # in meters
DISTANCE_FROM_CAMERA = 1 # of the object from the camera in meters
RULER_SIZE = 1 # in meters
ANGLE_OF_VIEW = np.arctan(RULER_SIZE / (2 * DISTANCE_FROM_CAMERA))

IMAGE = 1

img1 = cv.imread(f'images/right_img{IMAGE}.jpg')  # right image
img2 = cv.imread(f'images/left_img{IMAGE}.jpg')  # left image

if IMAGE == 2:
    # Cutoff for img2
    img1 = img1[200:900, 300:1000]
    img2 = img2[200:900, 300:1000]
# elif IMAGE == 3:
#     # Cutoff for img3
#     img1 = img1[200:500, 200:800]
#     img2 = img2[200:500, 200:800]


# normalize_factor = (DISTANCE_BETWEEN_CAMERAS * img1.shape[1]) / (2 * np.tan(ANGLE_OF_VIEW / 2))
normalize_factor = 1
# print(normalize_factor)

draw_distance_map(img1, img2, 11, normalize_factor=normalize_factor, cost="ssd", alg="regular", write=True, disp=True)

# Draw the epipolar lines on the right image and the original point on the left image
# pixel = (70, 80)
# x_values = range(10, len(img1[0])-11)
# line = get_parallel_line(pixel, x_values)
# sad_test(img1, img2, pixel, line, 11)
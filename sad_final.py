import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from epipolar_lines import get_parallel_line


def sad(point_list: list[tuple], point: tuple, image1: np.array, image2: np.array, size: int):
    """

    :param point_list: the line in which we want to find the point
    :param point: the point we want to find
    :param image1: the origin picture
    :param image2: the picture in which we are looking
    :param size: the size of neighborhood in which we are looking
    :return: the position (tuple) of the best match
    """
    # Check for edge cases
    if point[0] > len(image1) or point[1] > len(image1[0]):
        return None
    # Create
    padded_image1 = np.pad(image1, size, mode='constant', constant_values=0)
    block1 = padded_image1[point[0]:point[0] + 2 * size + 1, point[1]: point[1] + 2 * size + 1]
    padded_image2 = np.pad(image2, size, mode='constant', constant_values=0)

    min_sad_value = np.inf
    sad_lst = []
    # Run on all Points in the list
    for position in point_list:
        x, y = position[1], position[0]
        # Create new neighborhood
        block2 = padded_image2[y:y + 2 * size + 1, x:x + 2 * size + 1]
        sad_val = np.sum(np.abs(block1 - block2))
        if len(sad_lst) < 5:
            sad_lst.append((y, x, sad_val))
            sad_lst.sort(key=lambda a: a[2])
            min_sad_value = sad_lst[-1][-1]
        elif sad_val < min_sad_value:
            sad_lst.append((y, x, sad_val))
            sad_lst.sort(key=lambda a: a[2])
            min_sad_value = sad_lst[-1][-1]
            sad_lst.pop(0)
    sad_lst = [x[:2] for x in sad_lst]
    return sad_lst


def sad_test(img1, img2, pixel, line, size):
    img1 = cv.circle(img1, (pixel[1], pixel[0]), 3, (255, 0, 0), -1)
    matching_pixels = sad(line, pixel, img1, img2, size)
    print(matching_pixels)
    for i, matching_pixel in enumerate(matching_pixels):
        img2 = cv.circle(img2, (matching_pixel[1], matching_pixel[0]), 3, (255, i * 40, 0), -1)
    plt.subplot(121), plt.imshow(img1)
    plt.subplot(122), plt.imshow(img2)
    plt.show()


if __name__ == "__main__":
    lst2 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]])
    lst1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    point1 = (1, 1)
    point_lst = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]
    size = 5

    image1 = cv.imread('images/left_img_in.png', cv.IMREAD_GRAYSCALE)
    image2 = cv.imread('images/right_img_in.png', cv.IMREAD_GRAYSCALE)
    line = get_parallel_line((0, 0), range(0, len(image1[0])))

    print(sad(line, (0, 0), image1, image2, size))


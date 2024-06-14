import cv2
import numpy as np
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
    if point[0] > len(image1[0]) or point[1] > len(image1):
        return None

    padded_image1 = np.pad(image1, size, mode='constant', constant_values=0)
    block1 = padded_image1[point[1]:point[1] + 2 * size + 1, point[0]: point[0] + 2 * size + 1]
    padded_image2 = np.pad(image2, size, mode='constant', constant_values=0)

    min_sad_value = np.inf
    min_sad_pos = (0, 0)
    for position in point_list:
        x = position[1]
        y = position[0]

        block2 = padded_image2[y:y + 2 * size + 1, x:x + 2 * size + 1]
        sad_val = np.sum(np.abs(block1 - block2))

        if sad_val < min_sad_value:
            min_sad_value = sad_val
            min_sad_pos = (x, y)

    return min_sad_pos


if __name__ == "__main__":
    lst2 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]])
    lst1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    point1 = (1, 1)
    point_lst = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]
    size = 5

    image1 = cv2.imread('images/left_img_in.png', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread('images/right_img_in.png', cv2.IMREAD_GRAYSCALE)
    line = get_parallel_line((0, 0), range(0, len(image1[0])))

    print(sad([(0, 383)], (0, 0), image1, image2, size))


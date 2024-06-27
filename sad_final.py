import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from epipolar_lines import get_parallel_line


def sad(point_list: np.array, point: tuple, image1: np.array, image2: np.array, size: int, point_count: int = 8, blacklist: set[tuple] = None, cost = "sad"):
    """
    Computes the sum of absolute value of the differences
    :param point_list: the line in which we want to find the point
    :param point: the point we want to find
    :param image1: the origin picture
    :param image2: the picture in which we are looking
    :param size: the size of neighborhood in which we are looking
    :param blacklist: points that we already matched and dont want to match again
    :return: the position (tuple) of the best match
    """
    window_size = int(size/2)
    # Check for edge cases
    if point[0] > len(image1) or point[1] > len(image1[0]):
        return None
    block1 = image1[point[0]-window_size:point[0] + window_size + 1, point[1]-window_size: point[1] + window_size + 1]
    sad_lst = np.zeros((max(point_count, len(point_list)), 3))
    sad_lst[:, 2] = np.inf
    for i, position in enumerate(point_list):
        # if blacklist and tuple(position) in blacklist:
        #     continue
        x, y = position[1], position[0]
        # Create new neighborhood
        block2 = image2[y - window_size:y + window_size + 1, x - window_size: x + window_size + 1]
        sad_val = difference_cost(block1, block2)
        sad_lst[i] = np.array([y, x, sad_val])
    sorted_indices = np.argsort(sad_lst[:, 2])
    sad_lst = sad_lst[sorted_indices][:, :-1]
    return sad_lst.astype(int)[:point_count]


def difference_cost(image1, image2, algorithm: str = "sad"):
    """
    Gets two images and calcalates the difference cost between them using different algorithms
    :param image1:
    :param image2:
    :return:
    """
    if algorithm == "sad":
        return np.sum(np.abs(image1 - image2))
    elif algorithm == "ssd":
        return np.sum(np.square(image1 - image2))
    else:
        print("Invalid Algorithm")
        return None


def average_sad(point: tuple, image1: np.array, image2: np.array, size: int, search_window: int, threshold: int, algorithm: str):
    """
    Calculates the best match in img2 of points in point list to the point given in img1 using an averaging sad algorithm
    :param point: (y,x) point in image1 to find a match for
    :param image1: np array of cv left image
    :param image2: np array of cv right image
    :param size: size of the neighborhood to search
    :param search_window: how far around the point to search
    :param threshold: the threshold to calculate the disparity of a point
    :return: (y,x) of the best match
    """
    y, x0 = point
    # Check for edge cases
    if y > len(image1) or x0 > len(image1[0]):
        return None
    # Create
    block1 = image1[point[0] - size:point[0] + size + 1, point[1] - size: point[1] + size + 1]
    disparities = []
    total_difference = 0
    # Run on all Points in the list
    start = max(size, x0 - search_window)
    end = min(len(image1[0])-size, x0 + search_window)
    for x in range(start, end):
        # Check if the point passes the threshold
        if np.abs(image1[point[0], point[1]] - image2[y, x]) > threshold:
            continue
        # Create new neighborhood
        block2 = image2[y - size:y + size + 1, x - size: x + size + 1]
        difference = difference_cost(block1, block2, algorithm)
        disparities.append((x-x0) * difference)
        total_difference += difference
    if total_difference == 0:
        return y, x0
    avg_disparity = int(np.sum(disparities) / total_difference)
    return y, x0 + avg_disparity


def sad_test(img1, img2, pixel, line, size):
    matching_pixels = sad(line, pixel, img1, img2, size)[1]

    img1 = cv.circle(img1, (pixel[1], pixel[0]), 3, (255, 0, 0), -1)
    for i, matching_pixel in enumerate(matching_pixels):
        img2 = cv.circle(img2, (matching_pixel[1], matching_pixel[0]), 3, (255, i * 25, i * 25), -1)
    plt.subplot(121), plt.imshow(img1)
    plt.subplot(122), plt.imshow(img2)
    plt.show()


def average_sad_test(img1, img2, pixel, size, search_window, threshold, algorithm):
    gray_img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray_img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    matching_pixel = average_sad(pixel, gray_img1, gray_img2, size, search_window, threshold, algorithm)
    img1 = cv.circle(img1, (pixel[1], pixel[0]), 3, (255, 0, 0), -1)
    img2 = cv.circle(img2, (matching_pixel[1], matching_pixel[0]), 3, (255, 0, 0), -1)
    plt.subplot(121), plt.imshow(img1)
    plt.subplot(122), plt.imshow(img2)
    plt.show()


if __name__ == "__main__":
    pass


# PREVIOUS SAD CODE
# min_sad_value = np.inf
# sad_lst = []
#
# # Run on all Points in the list
# for position in point_list:
#     x, y = position[1], position[0]
#     # Create new neighborhood
#     block2 = image2[y - size:y + size + 1, x - size: x + size + 1]
#     sad_val = difference_cost(block1, block2)
#     if len(sad_lst) < 8:
#         sad_lst.append(np.array([y, x, sad_val]))
#         sad_lst.sort(key=lambda a: a[2])
#         min_sad_value = sad_lst[-1][-1]
#     elif sad_val < min_sad_value:
#         sad_lst.append(np.array([y, x, sad_val]))
#         sad_lst.sort(key=lambda a: a[2])
#         min_sad_value = sad_lst[-1][-1]
#         sad_lst.pop()
# sad_lst = np.array(sad_lst)[:, :-1]

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def sad(point: tuple, image1: np.array, image2: np.array, search_range: int, size: int, position="r", cost="sad"):
    """
    Computes the sum of absolute value of the differences
    :param point: the point we want to find
    :param image1: the origin picture
    :param image2: the picture in which we are looking
    :param search_range: the range in which we are looking
    :param size: the size of neighborhood in which we are looking
    :param cost: the cost function to use
    :param position: r is image1 is the right image else l
    :return: the position (tuple) of the best match
    """
    window_size = int(size/2)
    # Check for edge cases
    if point[0] > len(image1) or point[1] > len(image1[0]):
        return None

    x, y = point[1], point[0]
    dir = 1 if position == "r" else -1

    block1 = image1[point[0]-window_size:point[0] + window_size + 1, point[1]-window_size: point[1] + window_size + 1]
    sad_lst = np.zeros((search_range, 3))
    sad_lst[:, 2] = np.inf

    for disparity in range(search_range):
        if x + dir * disparity + window_size >= len(image1[0]):
            continue
        # Create new neighborhood
        block2 = image2[y - window_size:y + window_size + 1, x + dir * disparity - window_size: x + dir * disparity + window_size + 1]
        sad_val = difference_cost(block1, block2, cost)
        sad_lst[disparity] = np.array([y, x + dir * disparity, sad_val])

    sorted_indices = np.argsort(sad_lst[:, 2])
    sad_lst = sad_lst[sorted_indices][:, :-1]
    return sad_lst.astype(int)[0]


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
    elif algorithm == "cc":
        return np.sum((image1 - np.mean(image1)) * (image2 - np.mean(image2))) / (np.std(image1) * np.std(image2) * image1.size)
    else:
        print("Invalid Algorithm")
        return None


def sad_test(img1, img2, pixel, line, size):
    matching_pixels = sad(line, pixel, img1, img2, size, cost="cc")

    img1 = cv.circle(img1, (pixel[1], pixel[0]), 1, (255, 0, 0), -1)
    for i, matching_pixel in enumerate(matching_pixels):
        img2 = cv.circle(img2, (matching_pixel[1], matching_pixel[0]), 1, (255, i * 25, i * 25), -1)
    plt.subplot(121), plt.imshow(img1)
    plt.subplot(122), plt.imshow(img2)
    plt.show()


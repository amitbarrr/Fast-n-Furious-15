import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pickle

from epipolar_lines import get_parallel_line
from sad_final import sad


SEARCH_AREA = 100 # in pixels


def get_disparity(img1, img2, size):
    """
    Get the disparity map between two images.
    :param img1: left image. cv grayscale image
    :param img2: right image. cv grayscale image
    :param size: size of the neighborhood to search
    :return: disparity map
    """
    disparity_map = np.zeros(img1.shape)
    for row in range(len(img1)):
        last_x_value = 0
        print(row)
        for col in range(len(img1[row])):
            pixel = (row, col)
            sup = col + SEARCH_AREA if col + SEARCH_AREA < len(img1[row]) else len(img1[row])
            line = get_parallel_line(pixel, range(0, sup))
            matching_pixels = sad(line, pixel, img1, img2, size)
            if len(matching_pixels) > 0:
                matching_pixel = matching_pixels[0]
                disparity_map[row][col] = np.abs(pixel[1] - matching_pixel[1])
                # print(disparity_map[row][col], pixel[1], matching_pixel[1])
                last_x_value = matching_pixel[1]
    return disparity_map


def get_distance(disparity, normalize_factor):
    """
    Get the distance between the camera and the object
    :param disparity: disparity map of image. 2d np array
    :param normalize_factor: normalize factor calculated by trigonometry
    :return: distance
    """
    if np.any(disparity == 0):
        return np.zeros(disparity.shape)
    return normalize_factor / disparity


def cross_reference(image1, image2, line: list, five_points: list, point: tuple, size: int) -> tuple:
    """

    :param image1: image which is the origin of the five points
    :param image2: image which is the origin of the single point.
    :param line: the line in which the points reside
    :param five_points: the five best matches for the points using sad
    :param point:
    :return: best matching point
    """
    minimum = np.inf
    min_index = 0

    for match_point in five_points:
        checked_points = sad(line, match_point, image1, image2, size)

        difference = np.linalg.norm(checked_points - point, axis=1)
        best_match = np.min(difference)

        if best_match < minimum:
            minimum = best_match
            min_index = np.argmin(difference)

    return five_points[min_index]


def draw_distance_map(img1, img2, normalize_factor, size):
    """
    Draw the distance map of objects in the scene.
    :param img1: left image. opencv image
    :param img2: right image. opencv image
    :param normalize_factor: normalize factor calculated by trigonometry
    :param size: size of the neighborhood to search
    :return: None
    """
    gray_img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray_img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    # disparity_map = get_disparity(gray_img1, gray_img2, size)
    # with open("disparity_map.pkl", "wb") as f:
    #     pick = pickle.dumps(disparity_map)
    #     f.write(pick)
    with open("disparity_map.pkl", "rb") as f:
        disparity_map = pickle.loads(f.read())
    print(disparity_map)
    distance_map = np.array(list(map(lambda x: get_distance(x, normalize_factor), disparity_map)))
    cv.imwrite("distance_map.png", distance_map)
    plt.subplot(121), plt.imshow(img1)
    plt.subplot(122), plt.imshow(disparity_map, cmap='gray')
    plt.show()

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pickle

from sad_final import *
from patch_finding import find_patch_in_img

# SEARCH_AREA = 100  # How many pixels away from the current pixel to search
OFFSET = 30  # How many pixels at the start of the image to ignore


def get_disparity(img1, img2, size, cost):
    """
    Get the disparity map between two images.
    :param img1: left image. cv grayscale image
    :param img2: right image. cv grayscale image
    :param size: size of the neighborhood to search
    :param cost: the cost algorithm to use. sad or ssd
    :return: disparity map
    """
    disparity_map = np.zeros(img1.shape)
    blacklist = set()

    window_size = int(size/2)

    for row in range(window_size, len(img1) - window_size):
        print("Line: ", row)
        for col in range(window_size, len(img1[row]) - window_size):
            pixel = (row, col)
            sup = col + SEARCH_AREA if col + SEARCH_AREA < len(img1[row]) - window_size else len(img1[row]) - window_size - 1
            starting_point = max(OFFSET, col)
            line = get_parallel_line(pixel, range(starting_point, sup))
            matching_pixels = sad(line, pixel, img1, img2, size, blacklist=blacklist, cost=cost)
            # best_match = cross_reference(img1, img2, matching_pixels, pixel, size, cost=cost)
            best_match = 0
            if len(matching_pixels) > 0:
                matching_pixel = matching_pixels[best_match]
                disparity_map[row, col] = np.abs(pixel[1] - matching_pixel[1])
                # blacklist.add(tuple(matching_pixel))
    return disparity_map


def get_block_disparity(img1, img2, size):
    """
    Get the disparity map for blocks of given size between two images.
    :param img1: left image. cv grayscale image
    :param img2: right image. cv grayscale image
    :param size: size of the blocks to search
    :param cost: the cost algorithm to use. sad or ssd
    :return:
    """
    window_size = int(size/2)
    disparity_map = np.zeros(img1.shape)

    for row in range(window_size, len(img1) - window_size, 2 * window_size):
        print("Line: ", row)
        for col in range(window_size, len(img1[row]) - window_size, 2 * window_size):
            pixel = (row, col)
            block = img1[row - window_size:row + window_size, col - window_size:col + window_size]
            matching_pixel = find_patch_in_img(img2, block, pixel)[-1][0]
            disparity_map[row - window_size:row + window_size, col - window_size:col + window_size] = np.abs(pixel[1] - matching_pixel[1])
    return disparity_map


def get_average_block_disparity(img1, img2, size, cost):
    """
    Get the disparity map between two images.
    :param img1: left image. cv grayscale image
    :param img2: right image. cv grayscale image
    :param size: size of the neighborhood to search
    :param cost: the cost algorithm to use. sad or ssd
    :return: disparity map
    """
    disparity_map = np.zeros(img1.shape)
    blacklist = set()

    window_size = int(size/2)

    for row in range(window_size, len(img1) - window_size, 2 * window_size):
        print("Line: ", row)
        for col in range(window_size, len(img1[row]) - window_size, 2 * window_size):
            block_disparity = 0
            block_size = size ** 2
            for r_offset in range(-window_size, window_size):
                for c_offset in range(-window_size, window_size):
                    pixel = (row + r_offset, col + c_offset)
                    # sup = col + SEARCH_AREA if col + SEARCH_AREA < len(img1[row]) - window_size else len(img1[row]) - window_size - 1
                    sup = len(img1[row]) - window_size - 1
                    starting_point = max(OFFSET, col)
                    line = get_parallel_line(pixel, range(starting_point, sup))
                    matching_pixels = sad(line, pixel, img1, img2, size, blacklist=blacklist, cost=cost)
                    best_match = 0
                    if len(matching_pixels) > 0:
                        matching_pixel = matching_pixels[best_match]
                        block_disparity += np.abs(pixel[1] - matching_pixel[1])
            average_block_disparity = block_disparity / block_size
            disparity_map[row - window_size:row + window_size, col - window_size:col + window_size] = average_block_disparity
    return disparity_map


def get_average_disparity(img1, img2, size, search_window: int = 100, threshold: int = 20, algorithm: str = "sad"):
    """
        Get the disparity map between two images.
        :param img1: left image. cv grayscale image
        :param img2: right image. cv grayscale image
        :param size: size of the neighborhood to search
        :return: disparity map
        """
    window_size = int(size / 2)

    disparity_map = np.zeros(img1.shape)
    for row in range(size, len(img1) - window_size):
        print(row)
        for col in range(size, len(img1[row]) - window_size):
            pixel = (row, col)
            matching_pixel = average_sad(pixel, img1, img2, window_size, search_window, threshold, algorithm)
            disparity_map[row][col] = np.abs(pixel[1] - matching_pixel[1])
    return disparity_map

def get_distance(disparity, normalize_factor):
    """
    Get the distance between the camera and the object
    :param disparity: disparity map of image. 2d np array
    :param normalize_factor: normalize factor calculated by trigonometry
    :return: distance
    """
    return np.divide(normalize_factor, disparity, out=np.zeros_like(disparity), where=disparity != 0)


def cross_reference(image1, image2, five_points: list, point: tuple, size: int, search_range: int = 20, cost = "sad") -> int:
    """
    Gets five points in image2 matching to a point in image1, matches them back to image1 and returns the best match
    :param image1: image which is the origin of the five points
    :param image2: image which is the origin of the single point.
    :param five_points: the five best matches for the points using sad
    :param point: the point in image1 to match to
    :param size: size of the neighborhood to search
    :param search_range: how far to search for the best match
    :param cost: the cost algorithm to use. sad or ssd
    :return: best matching point
    """
    minimum = np.inf
    min_index = 0

    window_size = int(size/2)

    for p in five_points:
        y, x = p
        if y == x == 0:
            continue
        low = max(window_size, x - search_range)
        high = min(len(image1[0]) - window_size, x + search_range)
        line = get_parallel_line(point, range(low, high))
        checked_points = sad(line, p, image2, image1, size, cost=cost)

        difference = np.linalg.norm(checked_points - point, axis=1)
        best_match = np.min(difference)

        if best_match < minimum:
            minimum = best_match
            min_index = np.argmin(difference)

    return min_index


def draw_distance_map(img1, img2, size, normalize_factor=100, alg="regular", cost="sad", write=True, disp = True):
    """
    Draw the distance map of objects in the scene.
    :param img1: left image. opencv image
    :param img2: right image. opencv image
    :param normalize_factor: normalize factor calculated by trigonometry
    :param size: size of the neighborhood to search
    :param alg: algorithm to use. regular or average
    :param cost: the cost algorithm to use. sad or ssd
    :param write: write the disparity map to a file
    :return: None
    """
    gray_img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray_img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    if write:
        if alg == "regular":
            disparity_map = get_disparity(gray_img1, gray_img2, size, cost)
        elif alg == "average":
            disparity_map = get_average_disparity(gray_img1, gray_img2, size, 30, 20, cost)
        elif alg == "block":
            disparity_map = get_block_disparity(gray_img1, gray_img2, size)
        else:
            print("Invalid Algorithm")
            return
        with open("disparity_map.pkl", "wb") as f:
            pick = pickle.dumps(disparity_map)
            f.write(pick)
    else:
        with open("disparity_map.pkl", "rb") as f:
            disparity_map = pickle.loads(f.read())

    distance_map = get_distance(disparity_map, normalize_factor)
    cv.imwrite("distance_map.png", distance_map)
    plt.title(f"Size {size}")
    plt.subplot(121), plt.imshow(img1)
    if disp:
        plt.subplot(122), plt.imshow(disparity_map, cmap='gray_r')
    else:
        plt.subplot(122), plt.imshow(distance_map, cmap='gray_r')

    plt.show()

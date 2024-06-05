import numpy as np
import os
import typing
import cv2 as cv


def SAD(template: list, image: list, point: tuple[int, int]) -> int:
    """
    This function tries to assess how similar a point is to another point in two images, by using light intensities.
    :param template: A list representing the neighborhood of a certain pixel in the original image.
    :param image: The image where the point is located
    :param point: The point which we want to assess how similar it is.
    :return: a numba
    """
    match_value = 0
    point_x = point[0]
    point_y = point[1]
    top_x_range = max([len(template[0]), ])
    for i in range(len(template)):
        for j in range(len(template[i])):
            image_x = j + point_x
            image_y = i + point_y
            match_value += abs(template[i][j] - image[image_y][image_x])

    return match_value


def SAD_COMPLETE(template: list, image: list) -> tuple[int, int]:
    """
    This function runs the SAD algorithm on all points in an image and returns the minimum and the image.
    :param template:
    :param image:
    :return:
    """
    minimum = SAD(template, image, (0, 0))
    min_pos = (0, 0)
    for row in range(len(image) - len(template) + 1):
        for col in range(len(image[row]) - len(template[0]) + 1):
            match_value = SAD(template, image, (row, col))
            if match_value < minimum:
                minimum = match_value
                min_pos = (row, col)

    return min_pos


def sad(template, image):
    minimum = float('inf')
    for row in range(len(image) - len(template) + 1):
        for col in range(len(image[row]) - len(template[0]) + 1):


            # Compute the absolute differences
            absolute_differences = np.abs(image.astype(np.float32) - image1.astype(np.float32))

            # Sum the absolute differences
            sad = np.sum(absolute_differences)

            if sad < minimum:
                minimum = sad


if __name__ == "__main__":
    img_gray = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    val = SAD([[2, 3], [5, 6]], img_gray, (1, 0))
    val2 = SAD_COMPLETE([[2, 3], [5, 6]], img_gray)

    print(val2)

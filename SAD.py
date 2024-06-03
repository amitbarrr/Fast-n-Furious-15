import numpy as np
import os
import typing
import cv2 as cv


numba = int
img_gray = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


def SAD(template: list, image: list, point: tuple[int, int]) -> numba:
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
    for i in range(len(template)):
        for j in range(len(template[i])):
            image_x = j + point_x
            image_y = i + point_y
            if image_y < 0 or image_y > len(image) - 1:
                match_value += abs(template[i][j])
                continue
            if image_x < 0 or image_x > len(image[i]) - 1:
                match_value += abs(template[i][j])
                continue
            match_value += abs(template[i][j] - image[i + point_y][j + point_x])

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
    for row in range(len(image)):
        for col in range(len(image[row])):
            match_value = SAD(template, image, (row, col))
            if match_value < minimum:
                minimum = match_value
                min_pos = (row, col)

    return min_pos


if __name__ == "__main__":
    val = SAD([[2, 3], [5, 6]], img_gray, (1, 0))

    val2 = SAD_COMPLETE([[2, 3], [5, 6]], img_gray)

    print(val2)

from typing import Optional, List, Union
import copy
import math

import numpy as np

SingleChannelImage = list[list[int]]
ColoredImage = list[list[list[int]]]
Image = Union[ColoredImage, SingleChannelImage]
Kernel = list[list[float]]

SEARCH_AREA = 50

def bilinear_interpolation(image: SingleChannelImage, y: float, x: float) -> int:
    """
    Performs bilinear interpolation to estimate pixel values at non-integer coordinates,
    in order to get a finer resolution of detail of the image.

    Args:
        image (SingleChannelImage): The input image.
        y (float): The vertical coordinate.
        x (float): The horizontal coordinate.

    Returns:
        int: The interpolated pixel value.
    """

    height, width = len(image), len(image[0])
    dx = x - math.floor(x)
    dy = y - math.floor(y)
    right, left, up, down = min(math.ceil(x), width-1), math.floor(x), math.floor(y), min(math.ceil(y), height-1)
    top_left_pixel = image[up][left]
    top_right_pixel = image[up][right]
    bot_left_pixel = image[down][left]
    bot_right_pixel = image[down][right]
    return round(top_left_pixel * (1-dx) * (1-dy) + bot_left_pixel * dy * (1-dx) + top_right_pixel * dx * (1-dy) + bot_right_pixel * dx * dy)


def resize(image: SingleChannelImage, new_height: int, new_width: int) -> SingleChannelImage:
    """
    Resizes an image to a specified height and width using bilinear interpolation.

    Args:
        image (SingleChannelImage): The input image.
        new_height (int): The desired height of the resized image.
        new_width (int): The desired width of the resized image.

    Returns:
        SingleChannelImage: The resized image.
    """
    if len(image) == 0:
        return []
    height, width = len(image), len(image[0])
    new_image = [[0 for _ in range(new_width)] for _ in range(new_height)]
    for i in range(new_height):
        for j in range(new_width):
            relative_x = (j / (new_width-1)) * (width-1)
            relative_y = (i / (new_height-1)) * (height-1)
            new_image[i][j] = bilinear_interpolation(image, relative_y, relative_x)
    new_image[0][0], new_image[0][-1], new_image[-1][0], new_image[-1][-1] = image[0][0], image[0][-1], image[-1][0], image[-1][-1]
    return new_image


def rotate_90(image: Image, direction: str) -> Image:
    """
    Rotates an image by 90 degrees clockwise if given R as a direction or counterclockwise if given L.

    Args:
        image (Image): The input image.
        direction (str): The direction of rotation ("R" for clockwise, "L" for counterclockwise).

    Returns:
        Image: The rotated image.
    """
    height, width = len(image), len(image[0])
    rotated_image = [[0 for _ in range(height)] for _ in range(width)]
    if direction == "R":
        for i in range(height):
            for j in range(width):
                rotated_image[j][height - 1 - i] = image[i][j]
    elif direction == "L":
        for i in range(height):
            for j in range(width):
                rotated_image[width-1-j][i] = image[i][j]
    return rotated_image


def mean_square_error(A: SingleChannelImage, B: SingleChannelImage) -> float:
    """
    Calculates the mean square error between two images.

    Args:
        A (SingleChannelImage): The first image.
        B (SingleChannelImage): The second image.

    Returns:
        int: The mean square error between the two images.
    """
    height, width = len(A), len(A[0])
    square_error = 0
    for i in range(height):
        for j in range(width):
            square_error += (A[i][j] - B[i][j]) ** 2
    return square_error / (height * width)


def get_patch_in_image(image: SingleChannelImage, x: int, y: int, height: int, width: int) -> SingleChannelImage:
    """
    Gets a starting position and patch size and calculates the patch starting at the given position
    and of the given size. (given position is top left)

    Args:
        image (SingleChannelImage): The input image.
        x (int): The x-coordinate of the top-left corner of the patch.
        y (int): The y-coordinate of the top-left corner of the patch.
        height (int): The height of the patch.
        width (int): The width of the patch.

    Returns:
        SingleChannelImage: The extracted patch.
    """
    patch = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            patch[i][j] = image[y+i][x+j]
    return patch


def get_best_match(image: SingleChannelImage, patch: SingleChannelImage, center: tuple) -> tuple:
    """
    Finds the best match for a given patch within an image using mean square error.

    Args:
        image (SingleChannelImage): The input image.
        patch (SingleChannelImage): The patch to be matched.
        center (tuple): The pixel of the center of the patch in the image.

    Returns:
        tuple: A tuple containing the coordinates of the best match and the corresponding mean square error.
    """
    y, x = center

    height, width = len(image), len(image[0])
    patch_height, patch_width = len(patch), len(patch[0])
    min_mse = math.inf
    min_mse_patch = (0, 0)
    for j in range(width - patch_width + 1):
        current_patch = get_patch_in_image(image, j, y, patch_height, patch_width)
        current_mse = mean_square_error(patch, current_patch)
        if current_mse < min_mse:
            min_mse = current_mse
            min_mse_patch = (y, j)
    return min_mse_patch, min_mse


def get_best_match_in_area(image: SingleChannelImage, patch: SingleChannelImage, pixel: tuple) -> tuple:
    """
    Finds the best match for a given patch within a 3x3 area of an image using mean square error.

    Args:
        image (SingleChannelImage): The input image.
        patch (SingleChannelImage): The patch to be matched.
        pixel (tuple): The pixel coordinates within the image to search around.

    Returns:
        tuple: A tuple containing the coordinates of the best match and the corresponding mean square error.
    """
    height, width = len(image), len(image[0])
    patch_height, patch_width = len(patch), len(patch[0])
    min_mse = math.inf
    min_mse_patch = (0,0)
    for i in range(-1, 2):
        for j in range(-1, 2):
            x, y = pixel[0] + i, pixel[1] + j
            if not 0 <= y <= height - patch_height or not 0 <= x <= width - patch_width:
                continue
            current_patch = get_patch_in_image(image, x, y, patch_height, patch_width)
            current_mse = mean_square_error(patch, current_patch)
            if current_mse < min_mse:
                min_mse = current_mse
                min_mse_patch = (y, x)
    return min_mse_patch, min_mse


def find_patch_in_img(image, patch, center) -> list[tuple]:
    """
    Finds the best location within an image where a given patch is most likely to be found.

    Args:
        image (SingleChannelImage): The input image.
        patch (SingleChannelImage): The patch to be located.
        center (tuple): The pixel of the center of the patch in the image

    Returns:
        dict: A dictionary containing rotation angles as keys and corresponding best match coordinates and
         mean square errors as values.
    """
    height, width = len(image), len(image[0])

    patch_height, patch_width = len(patch), len(patch[0])
    pyramid_size = int(np.log(min(patch_width, patch_height)))
    pyramid = [(image, patch)]
    for i in range(1, pyramid_size):
        pyramid.append((resize(pyramid[-1][0], int(height / (2 ** i)), int(width / (2 ** i))),
                        resize(pyramid[-1][1], int(patch_height / (2 ** i)), int(patch_width / (2 ** i)))))
    pyramid = pyramid[::-1]
    best_locations = [get_best_match(pyramid[0][0], pyramid[0][1], center)]
    for i in range(1, pyramid_size):
        previous_pos = best_locations[-1][0]
        pixel_to_search = (previous_pos[1] * 2, previous_pos[0] * 2)
        best_locations.append((get_best_match_in_area(pyramid[i][0], pyramid[i][1], pixel_to_search)))
    return best_locations


if __name__ == '__main__':
    pass

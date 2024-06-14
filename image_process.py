from epipolar_lines import get_parallel_line
import numpy as np
import cv2 as cv

def get_disparity(img1, img2):
    """
    Get the disparity map between two images
    :param img1: left image
    :param img2: right image
    :return: disparity map
    """
    disparity_map = np.zeros(img1.shape)
    for row in range(len(img1)):
        last_pixel = (row, 0)
        for col in range(len(img1[row])):
            pixel = (row, col)
            line = get_parallel_line(pixel, range(last_pixel[1], len(img1[row])))
            # print(pixel, line)
            matching_pixel = (pixel[0], pixel[1] - 50)
            # matching_pixel = sad(img1, img2, pixel, line)
            disparity_map[row][col] = np.abs(pixel[1] - matching_pixel[1])
    return disparity_map


def get_distance(disparity, noramlize_factor):
    """
    Get the distance between the camera and the object
    :param disparity: disparity map
    :param noramlize_factor: normalize factor calculated by trigonometry
    :return: distance
    """
    return noramlize_factor / disparity


def draw_distance_map(img1, img2, normalize_factor):
    """
    Draw the distance map of objects in the scene
    :param img1: left image
    :param img2: right imageF
    :param normalize_factor: normalize factor calculated by trigonometry
    :return: None
    """
    disparity_map = get_disparity(img1, img2)
    distance_map = np.array(list(map(lambda x: get_distance(x, normalize_factor), disparity_map)))
    # distance_img = np.hstack((img1, distance_map))
    # distance_img = np.concatenate((img1, distance_map), axis=1)
    cv.imshow("Original Image", img1)
    cv.imshow("Distance Of Objects", distance_map)
    # cv.imshow("Distance Of Objects In Image", distance_img)
    cv.waitKey()

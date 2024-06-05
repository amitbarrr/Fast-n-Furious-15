import numpy as np


def get_epipolar_line(pixel, F, x_range):
    """
    Get the epipolar line given a point and the fundamental matrix
    :param pixel: (x, y)
    :param F: fundamental matrix np.array
    :return: epipolar line
    """
    p = np.array([pixel[0], pixel[1], 1])
    line_equation = np.dot(p, F)
    line = []
    print(line_equation)
    for x in x_range:
        # y = (-ax - c) / b
        y = int((-line_equation[0] * x - line_equation[2]) / line_equation[1])
        # for y in y_range:
        #     if np.abs(np.dot(line_equation, np.array([x, y, 1]))) < 0.5:
        #         line.append((x, y))
        line.append((y, x))
    return np.array(line)


def get_parallel_line(pixel, x_range):
    """
    Get the parallel line to the x-axis
    :param pixel: (x, y)
    :return: parallel line
    """
    line = []
    for x in range(x_range):
        line.append((pixel[1], x))
    return np.array(line)

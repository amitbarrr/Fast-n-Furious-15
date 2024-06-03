import numpy as np


def get_epipolar_line(pixel, F, x_range, y_range):
    """
    Get the epipolar line given a point and the fundamental matrix
    :param pixel: (x, y)
    :param F: fundamental matrix np.array
    :return: epipolar line
    """
    p = np.array([pixel[0], pixel[1], 1])
    line_equation = np.dot(p, F)
    line = []
    for x in x_range:
        for y in y_range:
            if np.dot(line_equation, np.array([x, y, 1])) < 1:
                line.append((x, y))
    return np.array(line)

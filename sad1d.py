import cv2
import numpy as np


def sad_1d(template: list, image, points: list) -> int:
    min = float('inf')
    for i in range(len(points)):
        match_value = 0
        for j in range(len(template)):
            for k in range(len(template[j])):
                match_value += abs(template[j][k] - image[points[j][1]][points[j][0]])

        if match_value < min:
            min = match_value

    return min


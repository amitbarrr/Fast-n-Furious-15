import pickle
from tqdm import tqdm

from sad_final import *
from patch_finding import get_best_match

SEARCH_AREA = 64  # How many pixels away from the current pixel to search
LEFT_RIGHT_FAIL_THRESHOLD = 10  # How many pixels away from the current pixel to search


def get_image_disparity(img1, img2, size, cost, position="r"):
    """
    Get the disparity map between two images.
    :param img1: left image. cv grayscale image
    :param img2: right image. cv grayscale image
    :param size: size of the neighborhood to search
    :param cost: the cost algorithm to use. sad or ssd
    :param position: is img1 the left of the right image
    :return: disparity map
    """
    other_position = "l" if position == "r" else "r"
    dir = 1 if position == "r" else -1

    pos_disparity_map = get_disparity_map(img1, img2, size, cost, position=position)
    other_disparity_map = get_disparity_map(img2, img1, size, cost, position=other_position)

    for row in range(1, len(pos_disparity_map)-1):
        for col in range(1, len(pos_disparity_map[row])-1):
            pos_disparity = pos_disparity_map[row, col]
            if np.abs(pos_disparity - other_disparity_map[row, int(col + dir * pos_disparity)]) > LEFT_RIGHT_FAIL_THRESHOLD:
                pos_disparity_map[row, col] = np.mean(
                    pos_disparity_map[row-1:row+2, col-1:col+2] * np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]))
    return pos_disparity_map

def get_disparity_map(img1, img2, size, cost, position="r"):
    """
    Get the disparity map between two images.
    :param img1: left image. cv grayscale image
    :param img2: right image. cv grayscale image
    :param size: size of the neighborhood to search
    :param cost: the cost algorithm to use. sad or ssd
    :param position: is img1 the left of the right image
    :return: disparity map
    """
    disparity_map = np.zeros(img1.shape)
    window_size = int(size/2)

    for row in range(window_size, len(img1) - window_size):
        print("Line: ", row)
        for col in range(window_size, len(img1[row]) - window_size):
            pixel = (row, col)
            matching_pixel = sad(pixel, img1, img2, SEARCH_AREA, size, position="r", cost=cost)
            disparity_map[row, col] = np.abs(pixel[1] - matching_pixel[1])
    return disparity_map


def get_block_disparity(img1, img2, size):
    """
    Get the disparity map for blocks of given size between two images.
    :param img1: left image. cv grayscale image
    :param img2: right image. cv grayscale image
    :param size: size of the blocks to search
    :return:
    """
    window_size = int(size/2)
    disparity_map = np.zeros(img1.shape)
    progress_bar = tqdm(total=len(range(window_size, len(img1) - window_size, 2 * window_size)))

    for row in range(window_size, len(img1) - window_size, 2 * window_size):
        progress_bar.update(1)
        for col in range(window_size, len(img1[row]) - window_size, 2 * window_size):
            pixel = (row, col)
            block = img1[row - window_size:row + window_size, col - window_size:col + window_size]
            matching_pixel = get_best_match(img2, block, pixel)[0]
            disparity_map[row - window_size:row + window_size, col - window_size:col + window_size] = np.abs(pixel[1] - matching_pixel[1])
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
        checked_points = sad(p, image2, image1, high-low, size, cost=cost)

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
            disparity_map = get_disparity_map(gray_img1, gray_img2, size, cost)
        elif alg == "leftright":
            disparity_map = get_image_disparity(gray_img1, gray_img2, size, cost)
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

    disparity_map = cv.normalize(disparity_map, disparity_map, 0, 255, norm_type=cv.NORM_MINMAX)
    disparity_map = np.uint8(disparity_map)
    distance_map = get_distance(np.float64(disparity_map), normalize_factor)
    distance_map = cv.normalize(distance_map, distance_map, 0, 255, norm_type=cv.NORM_MINMAX)
    distance_map = np.uint8(distance_map)
    distance_map = cv.applyColorMap(distance_map, cv.COLORMAP_JET)
    plt.title(f"Size {size}")
    cv.imshow("Right", img1)
    cv.imshow("Left", img2)
    cv.imshow("disparity", disparity_map)
    cv.imshow("distance", distance_map)
    cv.waitKey(0)

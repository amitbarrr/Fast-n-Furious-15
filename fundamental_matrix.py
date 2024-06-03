import numpy as np
import cv2 as cv


def get_keypoints(img):
    """
    Get keypoints and descriptors from an image
    :param img: opencv grayscale image
    """
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


def get_matches(img1, img2):
    """
    Get point matches between two images
    :param img1: opencv grayscale image
    :param img2: opencv grayscale image
    :return: points in img1, points in img2
    """
    kp1, des1 = get_keypoints(img1)
    kp2, des2 = get_keypoints(img2)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

    return pts1, pts2


def get_fundamental_matrix(img1, img2):
    """
    Get the fundamental matrix between two images
    :param img1: opencv grayscale image
    :param img2: opencv grayscale image
    :return: fundamental matrix, mask
    """
    pts1, pts2 = get_matches(img1, img2)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    pts_len = min(len(pts1), len(pts2))
    F, mask = cv.findFundamentalMat(pts1[:pts_len], pts2[:pts_len], cv.FM_LMEDS)
    return F, mask


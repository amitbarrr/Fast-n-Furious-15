import cv2
import numpy as np


def compute_ssd(left_block, right_block):
    """Compute the Sum of Squared Differences (SSD) between two blocks."""
    return np.sum(np.square((left_block - right_block)))


def stereo_match_ssd(left_img, right_img, block_size, num_disparities):
    """Compute disparity map using SSD block matching."""
    height, width = left_img.shape
    disparity_map = np.zeros((height, width), dtype=np.float32)

    half_block_size = block_size // 2

    for y in range(half_block_size, height - half_block_size):
        for x in range(half_block_size, width - half_block_size):
            min_ssd = float('inf')
            best_disparity = 0

            for d in range(num_disparities):
                if x - d - half_block_size < 0:
                    continue

                left_block = left_img[y - half_block_size:y + half_block_size + 1,
                             x - half_block_size:x + half_block_size + 1]
                right_block = right_img[y - half_block_size:y + half_block_size + 1,
                              x - half_block_size - d:x + half_block_size + 1 - d]

                ssd = compute_ssd(left_block, right_block)

                if ssd < min_ssd:
                    min_ssd = ssd
                    best_disparity = d

            disparity_map[y, x] = best_disparity

    return disparity_map


########################################################################################################################
########################################################################################################################
# SSD


# Load stereo images
left_img = cv2.imread('images/left_img6.jpg', cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread('images/right_img6.jpg', cv2.IMREAD_GRAYSCALE)

left_img = left_img[100:900, 0:700]
right_img = right_img[100:900, 0:700]

cv2.imshow('Left Image', left_img)
cv2.imshow('Right Image', right_img)
cv2.waitKey(0)


# Parameters
block_size = 5
num_disparities = 64

# Compute disparity map using SSD
disparity_map = stereo_match_ssd(left_img, right_img, block_size, num_disparities)

# # Normalize disparity map for visualization
disparity_map = cv2.normalize(disparity_map, disparity_map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_map = np.uint8(disparity_map)


# Display the results
cv2.imshow('Left Image', left_img)
cv2.imshow('Right Image', right_img)
cv2.imshow('Disparity Map', disparity_map)
cv2.waitKey(0)
cv2.destroyAllWindows()

########################################################################################################################
# SGBM?

def compute_matching_cost(left_img, right_img, block_size, num_disparities):
    height, width = left_img.shape
    matching_cost = np.zeros((height, width, num_disparities))

    half_block_size = block_size // 2

    for d in range(num_disparities):
        for y in range(half_block_size, height - half_block_size):
            for x in range(half_block_size, width - half_block_size):
                if x - d - half_block_size >= 0:
                    left_block = left_img[y - half_block_size:y + half_block_size + 1,
                                          x - half_block_size:x + half_block_size + 1]
                    right_block = right_img[y - half_block_size:y + half_block_size + 1,
                                            x - half_block_size - d:x + half_block_size + 1 - d]
                    ssd = np.sum((left_block - right_block) ** 2)
                    matching_cost[y, x, d] = ssd

    return matching_cost

def semi_global_matching(left_img, right_img, block_size, num_disparities, P1, P2):
    height, width = left_img.shape
    matching_cost = compute_matching_cost(left_img, right_img, block_size, num_disparities)

    # Cost aggregation along different directions
    directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    aggregated_cost = np.zeros_like(matching_cost)

    for dy, dx in directions:
        path_cost = np.zeros_like(matching_cost)
        for y in range(height):
            for x in range(width):
                for d in range(num_disparities):
                    if y - dy >= 0 and y - dy < height and x - dx >= 0 and x - dx < width:
                        cost = matching_cost[y, x, d]
                        prev_cost = path_cost[y - dy, x - dx, d]
                        min_prev_cost = np.min(path_cost[y - dy, x - dx, :])
                        path_cost[y, x, d] = cost + min(prev_cost, min_prev_cost + P1, np.min(path_cost[y - dy, x - dx, :] + P2))
                    else:
                        path_cost[y, x, d] = matching_cost[y, x, d]
        aggregated_cost += path_cost

    # Select disparity with minimum cost
    disparity_map = np.argmin(aggregated_cost, axis=2)

    return disparity_map

########################################################################################################################
# # # SGBM
#
#
# # Load and preprocess images
# left_img = cv2.imread('images/left_img4.jpg', cv2.IMREAD_GRAYSCALE)
# right_img = cv2.imread('images/left_img4.jpg', cv2.IMREAD_GRAYSCALE)
#
# # Parameters
# block_size = 5
# num_disparities = 64
# P1 = 8 * block_size ** 2
# P2 = 32 * block_size ** 2
#
# # Compute disparity map using simplified SGBM
# disparity_map = semi_global_matching(left_img, right_img, block_size, num_disparities, P1, P2)
#
# # Normalize and display disparity map
# disparity_map = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
# disparity_map = np.uint8(disparity_map)
#
# cv2.imshow('Disparity Map', disparity_map)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
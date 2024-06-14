import cv2
import numpy as np


def template_matching_sad(image, template):
    """
    Perform template matching using the Sum of Absolute Differences (SAD) algorithm.

    Parameters:
    - image: The input image (numpy array).
    - template: The template image (numpy array).

    Returns:
    - best_match_top_left: The top-left coordinates of the best match.
    - best_match_value: The SAD value of the best match.
    """
    # Get dimensions of the input image and the template
    image_height, image_width = image.shape
    template_height, template_width = template.shape

    # Initialize variables to store the best match coordinates and the minimum SAD value
    best_match_top_left = (0, 0)
    best_match_value = float('inf')

    # Slide the template over the input image
    for y in range(image_height - template_height + 1):
        for x in range(image_width - template_width + 1):
            # Extract the region of the input image that matches the current template position
            image_region = image[y:y + template_height, x:x + template_width]

            # Compute the SAD between the template and the current region
            sad = np.sum(np.abs(template.astype(np.float32) - image_region.astype(np.float32)))

            # Update the best match if the current SAD is less than the minimum SAD found so far
            if sad < best_match_value:
                best_match_value = sad
                best_match_top_left = (x, y)

    return best_match_top_left, best_match_value


# Example usage:
# Load the input image and template image in grayscale
image = cv2.imread('images/left_img.jpg', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('images/left_patch.png', cv2.IMREAD_GRAYSCALE)

if image is None or template is None:
    print("Error loading images")
else:
    best_match_top_left, best_match_value = template_matching_sad(image, template)
    print(f"Best match top-left coordinates: {best_match_top_left}")
    print(f"Best match SAD value: {best_match_value}")

    # Draw a rectangle around the best match area
    match_x, match_y = best_match_top_left
    match_width, match_height = template.shape[::-1]
    cv2.rectangle(image, (match_x, match_y), (match_x + match_width, match_y + match_height), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Template Matching Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
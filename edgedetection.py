import cv2
import numpy as np
import os

# Function to crop a circular region from the image
def crop_circle(image, radius_factor=0.85):
    # Get the dimensions of the image
    height, width = image.shape[:2]
    # Create a black image with the same dimensions
    mask = np.zeros((height, width), dtype=np.uint8)
    # Define the center and radius of the circle
    center = (int(width / 2), int(height / 2))
    original_radius = min(center[0], center[1])
    radius = int(original_radius * radius_factor)
    # Draw the circle on the mask
    cv2.circle(mask, center, radius, (255,), -1, cv2.LINE_AA)
    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

# Loop over the images
for i in range(1, 48):  # Adjust range according to the number of images
    path = f'kothmale apple-{i}.jpg'
    image = cv2.imread(path)

    # Check if the image was loaded correctly
    if image is None:
        print(f"Error: Image {path} cannot be loaded. Please check the path.")
        continue

    # Specify the new dimensions and resize the image
    resized_image = cv2.resize(image, (512, 512))

    # Crop the image to a circular region
    cropped_image = crop_circle(resized_image, radius_factor=0.8)
    output_path = f'output_{i}.jpg'
    cv2.imwrite(output_path, cropped_image)

    # Read the cropped image in grayscale
    img = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)

    # Normalize the image
    normalized_image = img / 255.0

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(normalized_image, (5, 5), 0)

    # Apply Otsu's thresholding
    _, otsu_thresholded_image = cv2.threshold((blurred_image * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Define a kernel for the morphological operations
    kernel = np.ones((6, 6), np.uint8)

    # Apply morphological closing to close small holes
    closed_image = cv2.morphologyEx(otsu_thresholded_image, cv2.MORPH_CLOSE, kernel)

    # Apply morphological opening to remove small holes
    filled_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, kernel)

    # Apply Canny edge detection
    canny_edges = cv2.Canny(filled_image, 100, 200)

    # Save the processed images
    cv2.imwrite(f'otsu_thresholded_image_{i}.jpg', otsu_thresholded_image)
    cv2.imwrite(f'closed_image_{i}.jpg', closed_image)
    cv2.imwrite(f'final_segmented_image_{i}.jpg', filled_image)
    cv2.imwrite(f'canny_edges_{i}.jpg', canny_edges)

# Example of displaying one set of processed images (from the last iteration)
cv2.imshow('Original Image', image)
cv2.imshow('Resized Image', resized_image)
cv2.imshow('Cropped Image', cropped_image)
cv2.imshow('Grayscale Image', img)
cv2.imshow('Blurred Image', blurred_image)
cv2.imshow('Otsu Thresholded Image', otsu_thresholded_image)
cv2.imshow('Closed Image', closed_image)
cv2.imshow('Final Segmented Image', filled_image)
cv2.imshow('Canny Edges', canny_edges)

# Wait for a key press and then close all windows
cv2.waitKey(0)


cv2.destroyAllWindows()

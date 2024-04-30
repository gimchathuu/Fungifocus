import cv2
import numpy as np

# Load the image
path = 'kothmale apple-41.jpg'
image = cv2.imread(path)

# Check if the image was loaded correctly
if image is None:
    print("Error: Image cannot be loaded. Please check the path.")
    exit()

# Specify the new dimensions
resized_image = cv2.resize(image, (512, 512))

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

cropped_image = crop_circle(resized_image, radius_factor=0.8)
cv2.imwrite('output.jpg', cropped_image)
# Read the image in grayscale
img = cv2.imread('output.jpg', cv2.IMREAD_GRAYSCALE)

# Normalize the image directly
normalized_image = img / 255.0
# Apply Gaussian blur to reduce noise and improve edge detection
blurred_image = cv2.GaussianBlur(normalized_image, (5, 5), 0)

# Apply Sobel operator in the X and Y directions
sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=5)  # Increased ksize
sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)  # Increased ksize

# Calculate the magnitude of the gradient
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
magnitude = np.uint8(magnitude)



# Display the images
cv2.imshow('Original Image', image)
cv2.imshow('Resized Image', resized_image)
cv2.imshow('Cropped Image', cropped_image)
cv2.imshow('Grayscale Image', img)
cv2.imshow('Blurred Image', blurred_image)
cv2.imshow('Sobel X', sobel_x)
cv2.imshow('Sobel Y', sobel_y)
cv2.imshow('Gradient Magnitude', magnitude)


# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

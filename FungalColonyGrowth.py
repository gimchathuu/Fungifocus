import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Function to crop a circular region from the image
def crop_circle(image, radius_factor=0.85):
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (int(width / 2), int(height / 2))
    original_radius = min(center[0], center[1])
    radius = int(original_radius * radius_factor)
    cv2.circle(mask, center, radius, (255,), -1, cv2.LINE_AA)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

# Function to measure colony size
def measure_colony_size(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Otsu's thresholding
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours of the segmented colony
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate area of the largest contour
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        colony_size = cv2.contourArea(max_contour)
    else:
        colony_size = 0
    
    return colony_size

# Lists to store colony sizes and image numbers
colony_sizes = []
image_numbers = []

# Loop over the images
for i in range(1, 48):  
    path = f'kothmale apple-{i}.jpg'
    image = cv2.imread(path)

    # Check if the image was loaded correctly
    if image is None:
        print(f"Error: Image {path} cannot be loaded. Please check the path.")
        continue

    # Crop the image to a circular region
    cropped_image = crop_circle(image, radius_factor=0.8)

    # Measure colony size
    colony_size = measure_colony_size(cropped_image)
    
    # Append colony size and image number to lists
    colony_sizes.append(colony_size)
    image_numbers.append(i)

# Plot the growth of the fungal colony
plt.plot(image_numbers, colony_sizes)
plt.xlabel('Image Number')
plt.ylabel('Colony Size')
plt.title('Fungal Colony Growth')
plt.grid(True)
plt.show()

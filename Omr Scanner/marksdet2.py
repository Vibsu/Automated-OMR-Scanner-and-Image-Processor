import cv2
import numpy as np
from sklearn.cluster import KMeans
import warnings

# Suppress FutureWarning for KMeans
warnings.simplefilter(action='ignore', category=FutureWarning)

grid_values = [
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
]


def identify_value(x, y, max_x, max_y, margin_left, margin_top, grid_width, grid_height):
    col = int((x - margin_left) / grid_width)
    row = int((y - margin_top) / grid_height)

    # Check if the bubble falls within the valid grid range
    if 0 <= row < len(grid_values) and 0 <= col < len(grid_values[0]):
        return grid_values[row][col]
    else:
        return None


def detect_bubbles_and_get_integrated_values2(image_path):
    # Load the color image
    img = cv2.imread(image_path)

    # Convert the image to grayscale for processing
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding to convert to binary image
    _, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Store the bubble positions of marked bubbles (those with black rectangle border)
    bubble_positions = []

    # Process each contour
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Filter out small noise contours (adjust the threshold as needed)
        if area < 100:
            continue

        # Draw bounding rectangle around each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Get the center position of the bubble
        center_x = x + w // 2
        center_y = y + h // 2

        # Check if the bubble is marked (has a black rectangle)
        bubble_positions.append([center_x, center_y])

    # Convert bubble positions to a NumPy array
    bubble_positions = np.array(bubble_positions)

    # Normalize bubble positions
    max_x, max_y = np.max(bubble_positions, axis=0)
    bubble_positions_normalized = bubble_positions / [max_x, max_y]

    # Apply K-Means clustering with 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=0)
    kmeans.fit(bubble_positions_normalized)
    cluster_labels = kmeans.predict(bubble_positions_normalized)

    # Find the region for the first part of 1/4th of the page (9x26 grid size with margins)
    margin_left = int(max_x * 0.633)  # 7% margin on the left (Move 1 inch to the left)
    margin_top = int(max_y * 0.723)   # 5% margin on the top
    margin_right = int(max_x * 0.01)  # 7% margin on the right (Move 1 inch to the right)
    margin_bottom = int(max_y * 0.19546)

    grid_width = int((max_x - margin_left - margin_right) / 10.7)
    grid_height = int((max_y - margin_top - margin_bottom) / 3.35)
    # Sort bubble positions based on their y-coordinate to ensure order
    bubble_positions_sorted = sorted(bubble_positions, key=lambda pos: pos[1])

    # Create a variable to store the row-wise values
    row_wise_values = ['0'] * 12  # Initialize with '0' for all 12 rows

    for i in range(len(bubble_positions_sorted)):
        x, y = bubble_positions_sorted[i]
        value = identify_value(x, y, max_x, max_y, margin_left, margin_top, grid_width, grid_height)

        # Append the value to the row-wise values list for the corresponding row
        if value is not None:
            row = int((y - margin_top) / grid_height)
            row_wise_values[row] = value

            # Draw the green rectangle for the marked bubble
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 255, 0), 2)
    
    # Concatenate the values and display the row-wise values
    subject_code_row_wise = " ".join(row_wise_values)
    #print(f"Detected Bubble Values (Row-Wise): {subject_code_row_wise}")

    # Draw the grid for the first cluster (cluster 0)
    for row in range(12):
        for col in range(12):
            # Draw the grid cells within the specified region
            grid_x1 = margin_left + col * grid_width
            grid_y1 = margin_top + row * grid_height
            grid_x2 = grid_x1 + grid_width
            grid_y2 = grid_y1 + grid_height
            cv2.rectangle(img, (grid_x1, grid_y1), (grid_x2, grid_y2), (0, 192, 203), 2)

    scale_percent = 30  # Adjust this value to control the scale
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    resized_img = cv2.resize(img, (width, height))

    # Display the image with detected bubbles and grid
    cv2.imshow('Detected Bubbles and Grid', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return subject_code_row_wise
if __name__ == "__main__":
    # Replace "your_image_path.jpg" with the path to your A4 size sheet booklet image
    image_path = "./gat3.jpg"
    detect_bubbles_and_get_integrated_values2(image_path)

import cv2
import numpy as np
from sklearn.cluster import KMeans

grid_values = [
    ['0', '0', 'A', 'A', 'A', '0', '0', '0', '0', '0'],
    ['1', '1', 'B', 'B', 'B', '1', '1', '1', '1', '9'],
    ['2', '2', 'C', 'C', 'C', '2', '2', '2', '2', 'J'],
    ['3', '3', 'D', 'D', 'D', '3', '3', '3', '3', 'T'],
    ['4', '4', 'E', 'E', 'E', '4', '4', '4', '4', '3'],
    ['5', '5', 'F', 'F', 'F', '5', '5', '5', '5', '3'],
    ['6', '6', 'G', 'G', 'G', '6', '6', '6', '6', '3'],
    ['7', '7', 'H', 'H', 'H', '7', '7', '7', '7', '3'],
    ['8', '8', 'I', 'I', 'I', '8', '8', '8', '8', '3'],
    ['9', '9', 'J', 'J', 'J', '9', '9', '9', '9', '3'],
    ['', '', 'K', 'K', 'K', '9', '0', '1', '2', '3'],
    ['', '', 'L', 'L', 'L', '9', '0', '1', '2', '3'],
    ['', '', 'M', 'M', 'M', '9', '0', '1', '2', '3'],
    ['', '', 'N', 'N', 'N', '9', '0', '1', '2', '3'],
    ['', '', 'O', 'O', 'O', '9', '0', '1', '2', '3'],
    ['', '', 'P', 'P', 'P', '9', '0', '1', '2', '3'],
    ['', '', 'Q', 'Q', 'Q', '9', '0', '1', '2', '3'],
    ['', '', 'R', 'R', 'R', '9', '0', '1', '2', '3'],
    ['', '', 'S', 'S', 'S', '9', '0', '1', '2', '3'],
    ['', '', 'T', 'T', 'T', '9', '0', '1', '2', '3'],
    ['', '', 'U', 'U', 'U', '9', '0', '1', '2', '3'],
    ['', '', 'V', 'V', 'V', '9', '0', '1', '2', '3'],
    ['', '', 'W', 'W', 'W', '9', '0', '1', '2', '3'],
    ['', '', 'X', 'X', 'X', '9', '0', '1', '2', '3'],
    ['', '', 'Y', 'Y', 'Y', '9', '0', '1', '2', '3'],
    ['', '', 'Z', 'Z', 'Z', '9', '0', '1', '2', '3']
]

def identify_value(x, y, max_x, max_y, margin_left, margin_top, grid_width, grid_height):
    col = int((x - margin_left) / grid_width)
    row = int((y - margin_top) / grid_height)

    # Check if the bubble falls within the valid grid range
    if 0 <= row < len(grid_values) and 0 <= col < len(grid_values[0]):
        return grid_values[row][col]
    else:
        return None

def detect_bubbles(image_path):
    # Load the image
    img = cv2.imread(image_path, 0)

    # Thresholding to convert to binary image
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image to draw the contours and clusters
    img_with_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

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
        cv2.rectangle(img_with_contours, (x, y), (x+w, y+h), (0, 255, 0), 2)

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

    # Draw clusters with different colors
    cluster_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
    for i, label in enumerate(cluster_labels):
        x, y = bubble_positions[i]
        color = cluster_colors[label]
        cv2.circle(img_with_contours, (x, y), 5, color, -1)

    # Find the region for the first part of 1/4th of the page (9x26 grid size with margins)
    margin_left = int(max_x * 0.07)  # 7% margin on the left (Move 1 inch to the left)
    margin_top = int(max_y * 0.13)   # 5% margin on the top
    margin_right = int(max_x * 0.58)  # 7% margin on the right (Move 1 inch to the right)
    margin_bottom = int(max_y * 0.30)

    grid_width = int((max_x - margin_left - margin_right) / 9)
    grid_height = int((max_y - margin_top - margin_bottom) / 26)

    # List to store the identified values of marked bubbles
    bubble_values = []

    # Find the grid cell for each bubble and identify the value
    column_wise_values = [[] for _ in range(9)]  # Create a list to store values for each column

    for i in range(len(bubble_positions)):
        x, y = bubble_positions[i]
        value = identify_value(x, y, max_x, max_y, margin_left, margin_top, grid_width, grid_height)
        if value is not None:
            bubble_values.append(value)

            # Store the value in the corresponding column's list
            col = int((x - margin_left) / grid_width)
            column_wise_values[col].append(value)

            # Draw the green rectangle for the marked bubble
            cv2.rectangle(img_with_contours, (x - 5, y - 5), (x + 5, y + 5), (0, 255, 0), 2)

            # Draw the blue rectangle for the identified grid cell
            col = int((x - margin_left) / grid_width)
            row = int((y - margin_top) / grid_height)
            grid_x1 = margin_left + col * grid_width
            grid_y1 = margin_top + row * grid_height
            grid_x2 = grid_x1 + grid_width
            grid_y2 = grid_y1 + grid_height
            cv2.rectangle(img_with_contours, (grid_x1, grid_y1), (grid_x2, grid_y2), (255, 0, 0), 2)

    # Concatenate the values and display the subject code
    subject_code = "".join(bubble_values)
    # print(f"Detected Subject Code: {subject_code}")

    # Integrate column-wise values into a single string
    integrated_values = "".join(["".join(values) for values in column_wise_values])
    #print(f"Integrated Values: {integrated_values}")
    
    # Resize the image for better visualization
    scale_percent = 30  # Adjust this value to control the scale
    width = int(img_with_contours.shape[1] * scale_percent / 100)
    height = int(img_with_contours.shape[0] * scale_percent / 100)
    resized_img = cv2.resize(img_with_contours, (width, height))

    # Display the image with detected contours and clustered bubbles
    cv2.imshow('Detected Contours and Clusters', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return integrated_values
if __name__ == "__main__":
    # Replace "your_image_path.jpg" with the path to your A4 size sheet booklet image
    image_path = "gat3.jpg"
    # usn=detect_bubbles_and_get_integrated_values(image_path)
    # print(f"USN: {usn}")
    detect_bubbles(image_path)

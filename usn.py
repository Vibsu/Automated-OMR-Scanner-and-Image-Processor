import cv2
import numpy as np
from sklearn.cluster import KMeans
from subcode import detect_bubbles
import warnings
#from marksdet import detect_bubbles_and_get_integrated_values1
grid_values = [
    ['0',  'A', 'A', '0','0','A', 'A', '0', '0', '0', '0'],
    ['1', 'B', 'B','1', '1','B', 'B', '1', '1', '1', '1'],
    ['2',  'C', 'C', '2','2','C', 'C', '2', '2', '2', 'J'],
    ['3',  'D', 'D','3', '3','D', 'D', '3', '3', '3', 'T'],
    ['4',  'E', 'E','4', '4','E', 'E', '4', '4', '4', '3'],
    ['5',  'F', 'F','5', '5','F', 'F', '5', '5', '5', '3'],
    ['6',  'G', 'G','6', '6','G', 'G', '6', '6', '6', '3'],
    ['7',  'H', 'H','7', '7','H', 'H', '7', '7', '7', '3'],
    ['8',  'I', 'I','8', '8','I', 'I', '8', '8', '8', '3'],
    ['9',  'J', 'J','9', '9','J', 'J', '9', '9', '9', '3'],
    ['', 'K', 'K','9', '9','K', 'K', '0', '1', '2', '3'],
    ['', 'L', 'L',' ', ' ','L', 'L', '0', '1', '2', '3'],
    ['', 'M', 'M',' ', ' ','M', 'M', '0', '1', '2', '3'],
    ['', 'N', 'N', ' ',' ','N', 'N', '0', '1', '2', '3'],
    ['', 'O', 'O',' ', ' ','O', 'O', '0', '1', '2', '3'],
    ['', 'P', 'P',' ', ' ','P', 'P', '0', '1', '2', '3'],
    ['', 'Q', 'Q',' ', ' ','Q', 'Q', '0', '1', '2', '3'],
    ['', 'R', 'R', ' ',' ','R', 'R', '0', '1', '2', '3'],
    ['', 'S', 'S',' ', ' ','S', 'S', '0', '1', '2', '3'],
    ['', 'T', 'T',' ', ' ','T', 'T', '0', '1', '2', '3'],
    ['', 'U', 'U',' ', ' ','U', 'U', '0', '1', '2', '3'],
    ['', 'V', 'V',' ', ' ','V', 'V', '0', '1', '2', '3'],
    ['', 'W', 'W',' ', ' ','W', 'W', '0', '1', '2', '3'],
    ['', 'X', 'X',' ', ' ','X', 'X', '0', '1', '2', '3'],
    ['', 'Y', 'Y',' ', ' ','Y', 'Y', '0', '1', '2', '3'],
    ['', 'Z', 'Z', ' ',' ','Z', 'Z', '0', '1', '2', '3'],
    
]

def identify_value(x, y, max_x, max_y, margin_left, margin_top, grid_width, grid_height):
    col = int((x - margin_left) / grid_width)
    row = int((y - margin_top) / grid_height)

    # Check if the bubble falls within the valid grid range
    if 0 <= row < len(grid_values) and 0 <= col < len(grid_values[0]):
        return grid_values[row][col]
    else:
        return None

def detect_bubbles_and_get_integrated_values(image_path):
    # Load the image
    img = cv2.imread(image_path, 0)
    integrated_values = detect_bubbles(image_path)
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
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Apply K-Means clustering with 4 clusters
    n_init = 10
    kmeans = KMeans(n_clusters=4, random_state=0, n_init=n_init)
    kmeans.fit(bubble_positions_normalized)
    cluster_labels = kmeans.predict(bubble_positions_normalized)
    
    # Draw clusters with different colors
    cluster_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
    for i, label in enumerate(cluster_labels):
        x, y = bubble_positions[i]
        color = cluster_colors[label]
        cv2.circle(img_with_contours, (x, y), 5, color, -1)

    # Find the region for the first part of 1/4th of the page (9x26 grid size with margins)
    margin_left = int(max_x * 0.6)  # 7% margin on the left (Move 1 inch to the left)
    margin_top = int(max_y * 0.13)   # 5% margin on the top
    margin_right = int(max_x * 0.06)  # 7% margin on the8right (Move 1 inch to the right)
    margin_bottom = int(max_y * 0.30)

    grid_width = int((max_x - margin_left - margin_right) / 8)
    grid_height = int((max_y - margin_top - margin_bottom) / 26)    # Draw the grid for the first cluster (cluster 0)
    for i, label in enumerate(cluster_labels):
        x, y = bubble_positions[i]
        color = cluster_colors[label]

        cv2.circle(img_with_contours, (x, y), 5, color, -1)

    # Draw grid only for the first cluster within the specified region
    if 1 in cluster_labels:
        for row in range(26):
            for col in range(10):
                # Draw the grid cells within the specified region
                grid_x1 = margin_left + col * grid_width
                grid_y1 = margin_top + row * grid_height
                grid_x2 = grid_x1 + grid_width
                grid_y2 = grid_y1 + grid_height
                cv2.rectangle(img_with_contours, (grid_x1, grid_y1), (grid_x2, grid_y2), (255, 0, 255), 2)


    bubble_values = []

    column_wise_values = [[] for _ in range(10)]  # Create a list to store values for each column

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
    #print(f"Detected USN: {subject_code}")
    

    # Integrate column-wise values into a single string
    integrated_values1 = "".join(["".join(values) for values in column_wise_values])
    #print(f"USN: {integrated_values1}")
    


    # Resize the image for better visualization
    scale_percent = 30  # Adjust this value to control the scale
    width = int(img_with_contours.shape[1] * scale_percent / 100)
    height = int(img_with_contours.shape[0] * scale_percent / 100)
    resized_img = cv2.resize(img_with_contours, (width, height))

    # Display the image with detected contours and clustered bubbles
    cv2.imshow('Detected Contours and Clusters', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return integrated_values1
if __name__ == "__main__":
    # Replace "your_image_path.jpg" with the path to your A4 size sheet booklet image
    image_path = "Gat.jpg"
    subj=detect_bubbles(image_path)
    print(f"Subject Code: {subj}")
    integrated_values_row_wise = detect_bubbles_and_get_integrated_values(image_path)
    

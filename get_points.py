
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to plot points on an image and get clicked coordinates
def get_clicked_points(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Click on the image to select points (press Enter to finish)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    points = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close()
    return points

# Load the frame image and top view image
frame_image = cv2.imread('src/3d_photo.jpg')
top_view_image = cv2.imread('src/top_view_court_img.png')

# Resize both images to 700x400 pixels
frame_image_resized = cv2.resize(frame_image, (700, 400))
top_view_image_resized = cv2.resize(top_view_image, (700, 400))

# Get clicked points on the frame image
clicked_points_frame = get_clicked_points(frame_image_resized)

# Get clicked points on the top view image
clicked_points_top_view = get_clicked_points(top_view_image_resized)

# Plot the clicked points on the frame image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(frame_image_resized, cv2.COLOR_BGR2RGB))
plt.title('Frame Image')
for point in clicked_points_frame:
    plt.plot(point[0], point[1], 'ro')  # 'ro' stands for red circles
    plt.text(point[0], point[1], f'({int(point[0])}, {int(point[1])})', fontsize=10, color='red')

# Plot the clicked points on the top view image
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(top_view_image_resized, cv2.COLOR_BGR2RGB))
plt.title('Top View Image')
for point in clicked_points_top_view:
    plt.plot(point[0], point[1], 'bo')  # 'bo' stands for blue circles
    plt.text(point[0], point[1], f'({int(point[0])}, {int(point[1])})', fontsize=10, color='blue')

plt.show()



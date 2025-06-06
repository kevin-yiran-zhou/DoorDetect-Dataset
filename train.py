#%%
import os
import cv2
import matplotlib.pyplot as plt
import random
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import shutil
import time


# Directory paths
images_dir = 'images'
labels_dir = 'labels'

# Load class names
with open('obj.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Define specific colors for each class
class_colors = {
    'door': (255, 140, 0),  # Dark Orange
    'handle': (0, 255, 0),  # Green
    'cabinet door': (255, 0, 0),  # Red
    'refrigerator door': (0, 0, 255)  # Blue
}



#%% 
# Function to visualize images with labels
def visualize_images_with_labels(num_images=5):
    image_files = os.listdir(images_dir)
    random.shuffle(image_files)
    image_files = image_files[:num_images]
    for image_file in image_files:
        # Read image
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read corresponding label
        label_file = image_file.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            labels = f.readlines()

        # Plot image
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(image_file)

        # Plot labels
        for label in labels:
            label_data = label.strip().split()
            # Assuming label format: class_id x_center y_center width height
            class_id, x_center, y_center, width, height = map(float, label_data)
            class_name = classes[int(class_id)]
            color = class_colors.get(class_name, (255, 255, 255))  # Default to white if class not found
            color = [c / 255 for c in color]  # Normalize color for Matplotlib
            # Convert to pixel values
            h, w, _ = image.shape
            x_center *= w
            y_center *= h
            width *= w
            height *= h
            # Calculate bounding box
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            # Draw rectangle
            plt.gca().add_patch(plt.Rectangle((x1, y1), width, height, edgecolor=color, facecolor='none', linewidth=2))
            # Add class label
            plt.text(x1, y1 - 10, class_name, color=color, fontsize=12, weight='bold')

        plt.show()

# Visualize some images
# visualize_images_with_labels()



# %%
# Check if train, val, and test directories exist
train_images_dir = 'data/train/images'
val_images_dir = 'data/val/images'
test_images_dir = 'data/test/images'
train_labels_dir = 'data/train/labels'
val_labels_dir = 'data/val/labels'
test_labels_dir = 'data/test/labels'

if not (os.path.exists(train_images_dir) and os.path.exists(val_images_dir) and os.path.exists(test_images_dir)):
    print("Splitting dataset into train, val, and test sets...")
    # Create directories
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)

    # Split the dataset
    image_files = os.listdir(images_dir)
    label_files = [f.rsplit('.', 1)[0] + '.txt' for f in image_files]
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        image_files, label_files, test_size=0.3, random_state=42
    )
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=0.5, random_state=42
    )

    # Copy files to respective directories
    for img, lbl in zip(train_images, train_labels):
        shutil.copy(os.path.join(images_dir, img), os.path.join(train_images_dir, img))
        shutil.copy(os.path.join(labels_dir, lbl), os.path.join(train_labels_dir, lbl))

    for img, lbl in zip(val_images, val_labels):
        shutil.copy(os.path.join(images_dir, img), os.path.join(val_images_dir, img))
        shutil.copy(os.path.join(labels_dir, lbl), os.path.join(val_labels_dir, lbl))

    for img, lbl in zip(test_images, test_labels):
        shutil.copy(os.path.join(images_dir, img), os.path.join(test_images_dir, img))
        shutil.copy(os.path.join(labels_dir, lbl), os.path.join(test_labels_dir, lbl))

    print("Datasets have been split and saved.")
else:
    print("Train, val, and test directories already exist.")

# Load images and labels from split directories
train_images = os.listdir(train_images_dir)
val_images = os.listdir(val_images_dir)
test_images = os.listdir(test_images_dir)
train_labels = os.listdir(train_labels_dir)
val_labels = os.listdir(val_labels_dir)
test_labels = os.listdir(test_labels_dir)

print(f"Training images: {len(train_images)}, Validation images: {len(val_images)}, Test images: {len(test_images)}")



#%%
# Load the pre-trained YOLOv8n model
model = YOLO('yolov8n.pt')

# Set up training parameters
train_params = {
    'data': 'dataset.yaml',  # Path to dataset configuration
    'epochs': 50,  # Number of epochs
    'batch': 16,  # Batch size
    'imgsz': 640,  # Image size
    'lr0': 0.01,  # Initial learning rate
}

# Train the model
# model.train(**train_params)
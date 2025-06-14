import os
import cv2
import matplotlib.pyplot as plt
import random
from ultralytics import YOLO
import time

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

# Directory paths
test_images_dir = 'data/collected'

# Load the best.pt model
model_path = 'runs/detect/train/weights/best.pt'
if os.path.exists(model_path):
    print("Loading best.pt model for evaluation...")
    model = YOLO(model_path)

    # Load images from test directory
    test_images = os.listdir(test_images_dir)

    # Define confidence score thresholds
    confidence_thresholds = {
        'door': 0.5,  # Example threshold for doors
        'handle': 0.3,  # Example threshold for door handles
        'cabinet door': 0.5,  # Example threshold for cabinet doors
        'refrigerator door': 0.5  # Example threshold for refrigerator doors
    }

    for img in test_images:
        img_path = os.path.join(test_images_dir, img)
        # Run inference
        start_time = time.time()
        result = model.predict(source=img_path, save=True, save_txt=True)
        inference_time = time.time() - start_time
        print(f"Inference time for {img}: {inference_time:.4f} seconds")

        # Read the image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract image dimensions
        h, w, _ = image.shape

        # Plot the image with predictions in a single window
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(f"Predictions for {img}")
        plt.axis('off')

        # Iterate over the list of results
        for res in result:
            # Access the boxes attribute for predictions
            boxes = res.boxes
            if boxes is not None:
                for box in boxes:
                    # Convert to pixel values and move to CPU
                    x_center, y_center, width, height = box.xywh[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = classes[class_id]
                    color = class_colors.get(class_name, (255, 255, 255))  # Default to white if class not found
                    color = [c / 255 for c in color]  # Normalize color for Matplotlib

                    # Ensure coordinates are normalized
                    x_center /= w
                    y_center /= h
                    width /= w
                    height /= h

                    # Convert to pixel values
                    x_center *= w
                    y_center *= h
                    width *= w
                    height *= h

                    # Calculate bounding box
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    # Extract confidence score
                    confidence = box.conf[0].cpu().numpy()
                    # Check if the confidence is above the threshold for the class
                    if confidence >= confidence_thresholds.get(class_name, 0.5):  # Default threshold if class not found
                        # Format the label with class name and confidence
                        label = f"{class_name} {confidence:.2f}"
                        # Add class label with confidence
                        plt.text(x1, y1 - 10, label, color=color, fontsize=12, weight='bold')
                        # Draw rectangle with thicker lines
                        plt.gca().add_patch(plt.Rectangle((x1, y1), width, height, edgecolor=color, facecolor='none', linewidth=3))

        plt.show()
else:
    print("best.pt model not found. Please ensure the model is saved correctly.") 
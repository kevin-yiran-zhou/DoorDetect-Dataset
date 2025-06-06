import cv2
import numpy as np
import time
import os
import random
import matplotlib.pyplot as plt
import cv2 as cv

# Load class names
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define specific colors for each class
class_colors = {
    'door': (255, 140, 0),  # Dark Orange
    'handle': (0, 255, 0),  # Green
    'cabinet door': (255, 0, 0),  # Red
    'refrigerator door': (0, 0, 255)  # Blue
}

# Load YOLO
net = cv2.dnn.readNet("yolo-obj.weights", "yolo-obj.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Set backend and target to use CPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load image
image_folder = "/home/kevin-zhou/Desktop/UMich/WeilandLab/DoorDetect-Dataset/images"  # Set your image folder path here
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
image_path = os.path.join(image_folder, random.choice(image_files))
img = cv.imread(image_path)
img = cv.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

# Start time
start_time = time.time()

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# End time
end_time = time.time()

# Calculate and print inference time
inference_time = end_time - start_time
print(f"Inference Time: {inference_time:.2f} seconds")

# Information to show on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-max suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Show the image in a larger window using Matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax = plt.gca()

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = class_colors.get(label, (255, 255, 255))  # Default to white if class not found
        color = [c / 255 for c in color]  # Normalize color for Matplotlib
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y - 10, f'{label} {confidence:.2f}', color=color, fontsize=12, weight='bold')

plt.axis('off')
plt.show()
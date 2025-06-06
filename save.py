import os
from ultralytics import YOLO

# Load the best.pt model
model_path = 'runs/detect/train/weights/best.pt'
if os.path.exists(model_path):
    print("Loading best.pt model for saving...")
    model = YOLO(model_path)
    # Export the model
    model.export(format='tflite', dynamic=False)
    print("Model exported to TFLite format.")

    # Export the model to ONNX format
    model.export(format='onnx', dynamic=False)
    print("Model exported to ONNX format.")
else:
    print("best.pt model not found. Please ensure the model is saved correctly.") 
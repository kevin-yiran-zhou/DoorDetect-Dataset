import os
from ultralytics import YOLO

# Load the best.pt model
# model_path = 'runs/detect/train/weights/best.pt'
# if os.path.exists(model_path):
#     print("Loading best.pt model for saving...")
#     model = YOLO(model_path)
#     # Export the model
#     model.export(format='tflite')
#     print("Model exported to TFLite format.")

#     # Export the model to ONNX format
#     model.export(format='onnx')
#     print("Model exported to ONNX format.")
# else:
#     print("best.pt model not found. Please ensure the model is saved correctly.") 



# Test exported models
# Test TFLite model
model_path = 'runs/detect/train/weights/best_saved_model/best_float16.tflite'
if os.path.exists(model_path):
    tflite_model = YOLO(model_path)
    results = tflite_model.predict(source="data/collected/IMG_AUSBC_20250606154746468.jpg", save=False)
    print(results)
else:
    print("TFLite model not found. Please ensure the model is saved correctly.")

# Test ONNX model
model_path = 'runs/detect/train/weights/best.onnx'
if os.path.exists(model_path):
    onnx_model = YOLO(model_path)
    results = onnx_model.predict(source="data/collected/IMG_AUSBC_20250606154746468.jpg", save=False)
    print(results)
else:
    print("ONNX model not found. Please ensure the model is saved correctly.")
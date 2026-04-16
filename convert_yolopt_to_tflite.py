from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom-trained model

# Export the model
model.export(format="tflite", int8=True, imgsz=320)

# Load a model
model = YOLO("yolo26s.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom-trained model

# Export the model
model.export(format="tflite", int8=True, imgsz=320)
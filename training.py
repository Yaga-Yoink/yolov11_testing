from ultralytics import YOLO

# Load a model
# -obb denotes oriented bounding box
# model = YOLO("yolo11n-obb.pt")  # load a pretrained model (recommended for training)

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Train the model
# results = model.train(data="dota8.yaml", epochs=20, imgsz=640, device= "mps")
# originally 50 epoch
results = model.train(data="video1_obb_data.yaml", epochs=1, imgsz=640)


# Save the model
# torch.save(model, "best.pt")

# Save to format used on drone

model.export(format = "onnx", opset=17)
# model.export(format = "imx")
# model.export(format = "tflite")
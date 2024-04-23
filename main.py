
from ultralytics import YOLO


# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

model.train(data="data.yaml", epochs=10)  # train the model

metrics = model.val()



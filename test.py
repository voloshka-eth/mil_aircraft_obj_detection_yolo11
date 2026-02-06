from ultralytics import YOLO

weights = r"/path/"

model = YOLO(weights)
model.info(detailed=True)

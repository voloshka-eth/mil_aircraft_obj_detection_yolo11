import torch
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
from ultralytics import YOLO


torch.serialization.add_safe_globals([DetectionModel, Sequential])

if __name__ == "__main__":

    model = YOLO('yolo11n.pt')

    results = model.train(
        data='Jet Detection/data.yaml',
        epochs=40,
        imgsz=960,
        batch=8,
        lr0=0.001,
        device=0,
        workers=4,
        project='jet_classifier',
        name='yolo11n_detection',
    )



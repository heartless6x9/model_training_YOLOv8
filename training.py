from ultralytics import YOLO

model = YOLO('yolov8x.pt')

model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    device='cpu'
)

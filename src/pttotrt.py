from ultralytics import YOLO

model = YOLO('model.pt', task='detect')

model.export(format='engine', simplify=True, imgsz=[640, 640], batch=1)

engine = YOLO('model.engine', task='detect')

results = engine(
    'https://news.cgtn.com/news/3d4d444e7759544f344d544e786b444d32556a4e31457a6333566d54/img/37dc7f636d794ca49bb8de7911606399/37dc7f636d794ca49bb8de7911606399.jpg')

print(results)

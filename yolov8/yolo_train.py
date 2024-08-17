"""Configure parameters and train YOLOv8."""


from ultralytics import YOLO


model = YOLO("yolov8m-cls.pt")
results = model.train(
    data='/home/pc0/projects/yolov7_training_old/data/plants/inat17_inat21_clefeol17_plantnet300k_gbif/',
    epochs=100, imgsz=224, device=0, batch=256, cos_lr=True,
    project='plants_classificator/yolov8/work_dir',
    name='inat17_inat21_clefeol17_plantnet300k_gbif_samples100_350_herb_25_100ep',
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, scale=0.2)

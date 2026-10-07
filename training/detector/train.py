from ultralytics import YOLO

def train_detection():
    model = YOLO("weights/yolov8n_detector.pt")
    # model = YOLO("yolov8n.yaml")
    result = model.train(
        data="./training/datasets/detector_dataset/data.yaml", batch=8
    )
    test = model.predict("pred_test.jpg")[0]
    test.save()


if __name__ == "__main__":
    train_detection()
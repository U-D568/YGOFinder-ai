from ultralytics import YOLO
from training.configs.mapper.detector import load_detector_config

def train_detection():
    config = load_detector_config()
    model = YOLO(str(config.model.pretrained_weights))
    # model = YOLO("yolov8n.yaml")
    result = model.train(
        data=str(config.data.dataset_yaml),
        batch=config.training.batch_size,
        epochs=config.training.epochs,
    )
    test = model.predict(str(config.prediction.image))[0]
    test.save()


if __name__ == "__main__":
    train_detection()
from ultralytics import YOLO

# Load a model
if __name__ == '__main__':
    model = YOLO(r"D:\Python project\yolo11\runs\train\autodl\exp-修改neckEL归一化200-\weights\best.pt")  # load a custom model
    data = "VisDrone-local.yaml"
    # Validate the model
    metrics = model.predict(
                        "D:\Python project\datasets\VisDrone\VisDrone2019-DET-test-dev/images",
                        device="0",
                        batch=16,
                        save=True,
                        conf=0.5,
                        )  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category
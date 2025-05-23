from ultralytics import YOLO

# Load a model
if __name__ == '__main__':
    model = YOLO(r"D:\Python project\yolo11\runs\train\comparison\yolov11n-HRSID-\weights\best.pt")  # load a custom model
    data = ("HRSID-test.yaml") # HRSID-test VisDrone-test.yaml
    # Validate the model
    metrics = model.val(
                        data=data,
                        imgsz=640,
                        device="0",
                        save_json=True,
                        project="runs/Test",
                        name="yolov11n-HRSID-",
                        plots=True,
                        conf=0.001,
                        exist_ok=False,
                        )  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category
from ultralytics import YOLO

# Load a model
if __name__ == '__main__':
    model = YOLO(r"D:\Python project\yolo11\runs\train\autodl\exp-SpdEL2-\weights\best.pt")  # load a custom model
    data = "VisDrone-local.yaml"
    # Validate the model
    metrics = model.val(
                        data=data,
                        device="0",
                        save_json=True,
                        project="runs/valid",
                        name="修改spd+检测头-",
                        plots=True,
                        conf=0.001,
                        exist_ok=False,
                        )  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category
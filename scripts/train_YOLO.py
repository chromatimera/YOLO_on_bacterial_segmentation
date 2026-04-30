from ultralytics import YOLO

def main():
    # 1. Load model (nano = fast, good for testing)
    model = YOLO("../yolov8n.pt")

    # 2. Train
    model.train(
        data="dataset.yaml",
        epochs=30,
        imgsz=1024,
        batch=4,          # reduce if memory issues
        project="runs",
        name="yolo_microscopy",
        verbose=True
    )

    # 3. Validate (optional but useful)
    metrics = model.val()
    print("Validation metrics:", metrics)

    # 4. Predict on validation images
    model.predict(
        source="dataset/images/val",
        save=True,
        project="runs",
        name="yolo_predictions"
    )

if __name__ == "__main__":
    main()
"""
Train a YOLO model on microscopy images.

Example usage:
python train_YOLO.py \
    --model yolov8n.pt \
    --data configs/dataset.yaml \
    --epochs 30

Note:
The dataset.yaml should point to the correct dataset root.
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def train_model(model_path, data_config, epochs, imgsz, batch, project, name):
    model = YOLO(model_path)

    model.train(
        data=str(data_config),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        verbose=True,
    )

    metrics = model.val()
    print("Validation metrics:", metrics)

    # Run predictions on validation images and save results
    model.predict(
        source="dataset/images/val",
        save=True,
        project=project,
        name=f"{name}_predictions",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLO model on a microscopy dataset."
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("yolov8n.pt"),
        help="Path to YOLO model weights (e.g. yolov8n.pt).",
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="Path to dataset YAML file.",
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)

    parser.add_argument(
        "--project",
        type=str,
        default="runs",
        help="Output directory for training runs.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="yolo_microscopy",
        help="Name of this training run.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train_model(
        model_path=args.model,
        data_config=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
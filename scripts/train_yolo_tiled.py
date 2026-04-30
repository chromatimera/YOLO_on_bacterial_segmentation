from ultralytics import YOLO

"""
Train a YOLO model on a tiled microscopy dataset.

Example usage:
python train_yolo_tiled.py \
    --model yolo11n.pt \
    --data configs/data_tiled.yaml \
    --imgsz 512 \
    --epochs 5 \
    --batch 8 \
    --name bacteria_tiled_512
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train_tiled_model(
    model_path,
    data_config,
    image_size,
    epochs,
    batch_size,
    patience,
    workers,
    project,
    name,
):
    model = YOLO(model_path)

    results = model.train(
        data=str(data_config),
        imgsz=image_size,
        epochs=epochs,
        batch=batch_size,
        patience=patience,
        workers=workers,
        project=project,
        name=name,
        pretrained=True,
        optimizer="auto",
        verbose=True,
    )

    best_model_path = Path(project) / name / "weights" / "best.pt"
    best_model = YOLO(best_model_path)

    metrics = best_model.val(
        data=str(data_config),
        imgsz=image_size,
        split="val",
    )

    print(metrics)
    return results, metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and validate a YOLO model on a tiled microscopy dataset."
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("yolo11n.pt"),
        help="Path to YOLO model weights, for example yolo11n.pt or yolov8n.pt.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/data_tiled.yaml"),
        help="Path to tiled dataset YAML config file.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=512,
        help="Training image size in pixels.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Output directory for YOLO training runs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="bacteria_tiled_512",
        help="Name of this training run.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train_tiled_model(
        model_path=args.model,
        data_config=args.data,
        image_size=args.imgsz,
        epochs=args.epochs,
        batch_size=args.batch,
        patience=args.patience,
        workers=args.workers,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
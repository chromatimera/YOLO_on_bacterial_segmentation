import argparse
import random
import shutil
from pathlib import Path

"""
WARNING:
This script MOVES files from train to validation.
Original training data will be modified.
"""

def split_dataset(dataset_path, val_fraction, seed):

    """
    Split YOLO dataset into train/val by moving a fraction of images and labels.
    Assumes YOLO folder structure:
    dataset/
        images/train
        labels/train
    """

    img_train = dataset_path / "images" / "train"
    lbl_train = dataset_path / "labels" / "train"

    img_val = dataset_path / "images" / "val"
    lbl_val = dataset_path / "labels" / "val"

    img_val.mkdir(parents=True, exist_ok=True)
    lbl_val.mkdir(parents=True, exist_ok=True)

    images = sorted(img_train.glob("*"))

    random.seed(seed)
    random.shuffle(images)

    n_val = int(val_fraction * len(images))
    val_images = images[:n_val]

    print(f"Moving {n_val} images to validation")

    for img_path in val_images:
        label_path = lbl_train / f"{img_path.stem}.txt"

        shutil.move(str(img_path), img_val / img_path.name)

        if label_path.exists():
            shutil.move(str(label_path), lbl_val / label_path.name)
        else:
            print(f"Warning: missing label for {img_path.name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Move a fraction of YOLO training images and labels into validation."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("../dataset"),
        help="Path to YOLO dataset folder.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of training images to move to validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    split_dataset(
        dataset_path=args.dataset,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
import numpy as np
import tifffile as tiff
from PIL import Image


def convert_images(dataset_path, splits):
    for split in splits:
        img_dir = dataset_path / "images" / split

        for img_path in list(img_dir.glob("*.tif")) + list(img_dir.glob("*.tiff")):
            img = tiff.imread(img_path)

            if img.ndim == 3:
                img = img[0]

            img = img.astype(np.float32)
            img = img - img.min()
            if img.max() > 0:
                img = img / img.max()

            img8 = (img * 255).astype(np.uint8)

            rgb = np.stack([img8, img8, img8], axis=-1)

            out_path = img_path.with_suffix(".png")
            Image.fromarray(rgb).save(out_path)

            img_path.unlink()
            print(f"Converted {img_path.name} -> {out_path.name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert TIFF microscopy images to PNG for YOLO training."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset"),
        help="Path to dataset root (must contain images/train, images/val, etc.).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to process (e.g. train val test).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    convert_images(dataset_path=args.dataset, splits=args.splits)


if __name__ == "__main__":
    main()
import os
import random
import shutil
from pathlib import Path

# paths
img_train = Path("../dataset/images/train")
lbl_train = Path("../dataset/labels/train")

img_val = Path("../dataset/images/val")
lbl_val = Path("../dataset/labels/val")

img_val.mkdir(parents=True, exist_ok=True)
lbl_val.mkdir(parents=True, exist_ok=True)

# get all images
images = list(img_train.glob("*"))

# shuffle
random.shuffle(images)

# select 20%
n_val = int(0.2 * len(images))
val_images = images[:n_val]

print(f"Moving {n_val} images to validation")

for img_path in val_images:
    label_path = lbl_train / (img_path.stem + ".txt")

    # move image
    shutil.move(str(img_path), img_val / img_path.name)

    # move label if exists
    if label_path.exists():
        shutil.move(str(label_path), lbl_val / label_path.name)
    else:
        print("Warning: missing label for", img_path.name)
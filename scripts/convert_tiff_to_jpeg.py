from pathlib import Path
import numpy as np
import tifffile as tiff
from PIL import Image

for split in ["train", "val"]:
    img_dir = Path(f"dataset/images/{split}")

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
from pathlib import Path
from PIL import Image

# ----------------------------
# Settings
# ----------------------------
INPUT_DATASET = Path("../dataset")
OUTPUT_DATASET = Path("../dataset_tiled")

TILE_SIZE = 512
STRIDE = 384   # 512 - 384 = 128 px overlap

SPLITS = ["train", "val"]


def read_yolo_labels(label_path, img_w, img_h):
    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, xc, yc, bw, bh = parts
            cls = int(cls)

            xc = float(xc) * img_w
            yc = float(yc) * img_h
            bw = float(bw) * img_w
            bh = float(bh) * img_h

            x1 = xc - bw / 2
            y1 = yc - bh / 2
            x2 = xc + bw / 2
            y2 = yc + bh / 2

            boxes.append([cls, x1, y1, x2, y2])

    return boxes


def box_intersection_with_tile(box, x0, y0, tile_size):
    cls, x1, y1, x2, y2 = box

    tx1, ty1 = x0, y0
    tx2, ty2 = x0 + tile_size, y0 + tile_size

    ix1 = max(x1, tx1)
    iy1 = max(y1, ty1)
    ix2 = min(x2, tx2)
    iy2 = min(y2, ty2)

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    original_area = (x2 - x1) * (y2 - y1)
    intersection_area = (ix2 - ix1) * (iy2 - iy1)

    # Keep box only if most of it is inside the tile
    if intersection_area / original_area < 0.5:
        return None

    # Convert to tile coordinates
    ix1 -= x0
    ix2 -= x0
    iy1 -= y0
    iy2 -= y0

    return [cls, ix1, iy1, ix2, iy2]


def convert_box_to_yolo(box, tile_size):
    cls, x1, y1, x2, y2 = box

    xc = ((x1 + x2) / 2) / tile_size
    yc = ((y1 + y2) / 2) / tile_size
    bw = (x2 - x1) / tile_size
    bh = (y2 - y1) / tile_size

    return f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def tile_split(split):
    image_dir = INPUT_DATASET / "images" / split
    label_dir = INPUT_DATASET / "labels" / split

    out_image_dir = OUTPUT_DATASET / "images" / split
    out_label_dir = OUTPUT_DATASET / "labels" / split

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(image_dir.glob("*.png")))

    for image_path in image_paths:
        img = Image.open(image_path).convert("RGB")
        img_w, img_h = img.size

        label_path = label_dir / f"{image_path.stem}.txt"
        boxes = read_yolo_labels(label_path, img_w, img_h)

        for y0 in range(0, img_h - TILE_SIZE + 1, STRIDE):
            for x0 in range(0, img_w - TILE_SIZE + 1, STRIDE):

                tile = img.crop((x0, y0, x0 + TILE_SIZE, y0 + TILE_SIZE))

                tile_boxes = []
                for box in boxes:
                    clipped = box_intersection_with_tile(box, x0, y0, TILE_SIZE)
                    if clipped is not None:
                        tile_boxes.append(clipped)

                # Optional: skip empty tiles
                if len(tile_boxes) == 0:
                    continue

                tile_name = f"{image_path.stem}_x{x0}_y{y0}.png"
                tile_label_name = f"{image_path.stem}_x{x0}_y{y0}.txt"

                tile.save(out_image_dir / tile_name)

                with open(out_label_dir / tile_label_name, "w") as f:
                    for box in tile_boxes:
                        f.write(convert_box_to_yolo(box, TILE_SIZE) + "\n")

        print(f"Done: {image_path.name}")


if __name__ == "__main__":
    for split in SPLITS:
        tile_split(split)

    print("Finished tiling dataset.")
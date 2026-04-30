from ultralytics import YOLO

# ----------------------------
# Settings
# ----------------------------
MODEL = "yolo11n.pt"        # or "yolov8n.pt"
DATA = "data_tiled.yaml"   # your tiled dataset config
IMG_SIZE = 512
EPOCHS = 5
BATCH = 8

# ----------------------------
# Train model
# ----------------------------
model = YOLO(MODEL)

results = model.train(
    data=DATA,
    imgsz=IMG_SIZE,
    epochs=EPOCHS,
    batch=BATCH,
    patience=30,
    workers=4,
    project="runs/detect",
    name="bacteria_tiled_512",
    pretrained=True,
    optimizer="auto",
    verbose=True,
)

# ----------------------------
# Validate best model
# ----------------------------
best_model = YOLO("runs/detect/bacteria_tiled_512/weights/best.pt")

metrics = best_model.val(
    data=DATA,
    imgsz=IMG_SIZE,
    split="val",
)

print(metrics)
import os
import shutil
import yaml
from ultralytics import YOLO

# Paths
dataset_path = "D:/computer_vision_project/datasets/datasets"
train_images = os.path.join(dataset_path, "cropped_images/train")
val_images = os.path.join(dataset_path, "cropped_images/val")
train_labels = os.path.join(dataset_path, "cropped_labels/train")
val_labels = os.path.join(dataset_path, "cropped_labels/val")

# Check if dataset directories exist
for path in [train_images, val_images, train_labels, val_labels]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Dataset path not found: {path}")

# Delete YOLO cache files if they exist
cache_files = [
    os.path.join(train_images, "train.cache"),
    os.path.join(val_images, "val.cache")
]
for cache in cache_files:
    if os.path.exists(cache):
        os.remove(cache)
        print(f"✅ Deleted cache: {cache}")

# Ensure YAML file is updated
ppe_config = {
    "path": dataset_path,
    "train": "cropped_images/train",
    "val": "cropped_images/val",
    "nc": 9,
    "names": [
        "hard-hat", "gloves", "mask", "glasses", "boots",
        "vest", "ppe-suit", "ear-protector", "safety-harness"
    ]
}

yaml_path = "ppe_config.yaml"
with open(yaml_path, "w") as file:
    yaml.dump(ppe_config, file, default_flow_style=False)
print(f"✅ Updated dataset YAML: {yaml_path}")

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Train model
results = model.train(
    data=yaml_path,
    epochs=25,
    imgsz=640,
    batch=8,
    workers=0
)

# Save trained model
runs_dir = "runs/detect/"
train_dirs = sorted(
    [d for d in os.listdir(runs_dir) if d.startswith("train") and d.replace("train", "").isdigit()],
    key=lambda x: int(x.replace("train", "")), reverse=True
)

if train_dirs:
    latest_run = train_dirs[0]
    best_model_path = os.path.join(runs_dir, latest_run, "weights", "best.pt")

    if os.path.exists(best_model_path):
        os.makedirs("weights", exist_ok=True)
        shutil.move(best_model_path, "weights/ppe_model.pt")
        print("\n✅ PPE detection model saved at 'weights/ppe_model.pt'")
    else:
        print("\n⚠️ Best model file not found. Please check training output.")
else:
    print("\n⚠️ No training runs found.")

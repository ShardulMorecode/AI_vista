#model  training(commented)
'''from ultralytics import YOLO

if __name__ == "__main__":
    # Load YOLOv8 model
    model = YOLO("yolov8s.pt")  

    # Train the model
    results = model.train(
        data="person_config.yaml", 
        epochs=25, 
        imgsz=640, 
        batch=4,   # Adjust batch size if needed
        workers=0  # Avoid multiprocessing issues on Windows
    )

    # Save trained model weights
    model.export(format="onnx")  # Optional: Export to ONNX format

    # Move best weights to a dedicated folder
    import shutil
    shutil.move("runs/detect/train/weights/best.pt", "weights/person_model.pt")

    print("\n✅ Training completed! Model saved at 'weights/person_model.pt'")
'''
#model saving
import os
import shutil

# Path to the saved model
saved_model_path = "weights/person_model.pt"

# Check if the model is already moved
if os.path.exists(saved_model_path):
    print("\n✅ Model already exists at 'weights/person_model.pt'. No need to move it again.")
else:
    # Find latest training folder
    runs_dir = "runs/detect/"
    train_dirs = sorted([d for d in os.listdir(runs_dir) if d.startswith("train")], key=lambda x: int(x.replace("train", "")), reverse=True)

    if train_dirs:
        latest_run = train_dirs[0]
        best_model_path = os.path.join(runs_dir, latest_run, "weights", "best.pt")

        if os.path.exists(best_model_path):
            os.makedirs("weights", exist_ok=True)
            shutil.move(best_model_path, saved_model_path)
            print("\n✅ Model saved at 'weights/person_model.pt'")
        else:
            print("\n⚠️ Best model file not found in the latest training folder.")
    else:
        print("\n⚠️ No training runs found.")

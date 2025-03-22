# AI Vision: PPE and Person Detection

This repository contains the **minimal version** of the **AI Vision** project, designed for detecting **Personal Protective Equipment (PPE)** and **persons** using a YOLO-based deep learning model. This version includes only the essential files for inference and prediction.

---

## Project Overview
The project leverages computer vision to:
- Detect individuals in images/videos.
- Detect PPE to ensure safety compliance.

This minimal version allows users to directly utilize the trained models for prediction without needing the full training datasets and scripts.

---

## Features
- **Object Detection**: Identify persons and PPE in real-time.
- **Pre-trained Models**: Includes weights for efficient inference.
- **Easy Configuration**: Modular YAML configuration for flexibility.

---

## Folder Structure
```plaintext
├── Documentation/         # Project documentation (optional).
├── datasets/              # Excluded in minimal version (available in full project).
├── weights/               # Trained model weights for inference.
├── README.md              # Project overview and setup instructions.
├── inference.py           # Script for running predictions.
├── pascalVOC_to_yolo.py   # (Optional) Data format conversion script.
├── pascalVOC_to_yolo_ppe.py   # Conversion script specific to PPE.
├── person_config.yaml     # Configuration for person detection.
├── ppe_config.yaml        # Configuration for PPE detection.
├── train_person.py        # Excluded in minimal version (training script).
├── train_ppe.py           # Excluded in minimal version (training script).
```

---

## Setup Instructions
Follow these steps to set up and use the minimal project version:

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/AI_vision.git
cd AI_vision
```

### 2. Install Dependencies
Ensure you have Python installed. Install the required packages using:
```bash
pip install -r requirements.txt
```

### 3. Run Inference
Use the `inference.py` script to perform predictions:
```bash
python inference.py --image_path <path_to_image> --config <path_to_config> --weights <path_to_weights>
```
Replace `<path_to_image>`, `<path_to_config>`, and `<path_to_weights>` with appropriate file paths.

---

## Full Project Access
The complete project, which includes the datasets, training scripts, and additional files, is available on Google Drive.

**Google Drive Link**: [Full Project](https://drive.google.com/file/d/1W2kJyW6lHnpAVqeB8dw41CRVTNRpGV8b/view?usp=drive_link)

---

## Acknowledgments
Special thanks to:
- **YOLO Framework**: For state-of-the-art object detection.
- **OpenCV**: For image and video processing.
- **PyTorch**: For building and training deep learning models.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Feel free to explore and modify this project as per your requirements! 🚀


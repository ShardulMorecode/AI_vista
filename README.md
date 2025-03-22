AI Vision: PPE and Person Detection
This repository contains the minimal version of the AI Vision project, designed for detecting personal protective equipment (PPE) and individuals using a YOLO-based deep learning model. The repository includes the trained model weights and essential files required to run inference and make predictions.

Project Overview
The goal of this project is to build a robust computer vision model capable of detecting:

Persons

Personal Protective Equipment (PPE)

Using advanced techniques and YOLO-based architecture, this system ensures accuracy and real-time performance in various applications, such as safety monitoring on construction sites.

Features
Object Detection: Detects individuals and PPE in images/videos.

Pre-trained Models: Trained weights for high-performance inference.

Configurable: Includes YAML configuration files for easy setup and modifications.

Folder Structure
pgsql
Copy
Edit
├── Documentation/         # Contains project-related documentation (optional).
├── datasets/              # (Excluded) Full dataset available via Google Drive link.
├── weights/               # Trained model weights for inference.
├── README.md              # Project details and setup instructions.
├── inference.py           # Script for making predictions.
├── pascalVOC_to_yolo.py   # (Optional) Script for data format conversion.
├── pascalVOC_to_yolo_ppe.py   # (Optional) Conversion script specific to PPE.
├── person_config.yaml     # Configurations for person detection.
├── ppe_config.yaml        # Configurations for PPE detection.
├── train_person.py        # (Excluded) Training script for person detection.
├── train_ppe.py           # (Excluded) Training script for PPE detection.
Setup Instructions
Follow the steps below to set up and run the project:

1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/AI_vision.git
cd AI_vision
2. Install Dependencies
Make sure you have Python installed on your system. Install the required libraries using:

bash
Copy
Edit
pip install -r requirements.txt
3. Run Inference
Use the inference.py script to make predictions:

bash
Copy
Edit
python inference.py --image_path <path_to_image> --config <path_to_config> --weights <path_to_weights>
Replace <path_to_image>, <path_to_config>, and <path_to_weights> with the appropriate paths.

Full Project Access
The complete project, including datasets, training scripts, and additional files, is available on Google Drive.
📂 Google Drive Link: Full Project

Acknowledgments
YOLO Framework: For its powerful object detection architecture.

OpenCV: For image and video processing.

PyTorch: For building and training the deep learning models.

License
This project is licensed under the MIT License. See the LICENSE file for details.

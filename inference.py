import os
import cv2
import torch
import numpy as np
import argparse
from ultralytics import YOLO

def load_model(model_path):
    """Load a YOLOv8 model."""
    return YOLO(model_path)

def detect_persons(model, image):
    """Detect persons in an image and return bounding boxes."""
    results = model(image)
    person_bboxes = []
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            if int(cls) == 0:  # Assuming '0' is the person class
                person_bboxes.append(box.cpu().numpy())
    return person_bboxes

def crop_persons(image, person_bboxes):
    """Crop detected persons from the full image."""
    cropped_images = []
    for bbox in person_bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cropped = image[y1:y2, x1:x2]
        cropped_images.append((cropped, (x1, y1, x2, y2)))
    return cropped_images

def detect_ppe(model, cropped_image):
    """Detect PPE items in a cropped person image."""
    results = model(cropped_image)
    detections = []
    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            detections.append((box.cpu().numpy(), int(cls), float(conf)))
    return detections

def draw_detections(image, person_bboxes, ppe_detections):
    """Draw bounding boxes on the image."""
    for bbox in person_bboxes:
        if len(bbox) != 4:
            print(f"❌ Invalid bbox: {bbox}")
            continue  

        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return image

def process_image(image_path, person_model, ppe_model, output_dir):
    """Process a single image through both models."""
    image = cv2.imread(image_path)
    person_bboxes = detect_persons(person_model, image)
    cropped_images = crop_persons(image, person_bboxes)
    ppe_detections = []
    
    for cropped, (x1, y1, x2, y2) in cropped_images:
        detections = detect_ppe(ppe_model, cropped)
        for det in detections:
            bbox, cls, conf = det
            bbox[0] += x1  # Adjust coordinates
            bbox[1] += y1
            bbox[2] += x1
            bbox[3] += y1
            ppe_detections.append((bbox, cls, conf))
    
    output_image = draw_detections(image, person_bboxes, ppe_detections)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, output_image)
    print(f"✅ Processed and saved: {output_path}")

def main(input_dir, output_dir, person_model_path, ppe_model_path):
    """Main function to run inference on all images."""
    person_model = load_model(person_model_path)
    ppe_model = load_model(ppe_model_path)

    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]

    for img_file in image_files:
        process_image(os.path.join(input_dir, img_file), person_model, ppe_model, output_dir)
    
    print("🎯 Inference completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on images.")
    parser.add_argument("input_dir", help="Path to the input directory containing images")
    parser.add_argument("output_dir", help="Path to save the output images")
    parser.add_argument("person_det_model", help="Path to the person detection model")
    parser.add_argument("ppe_detection_model", help="Path to the PPE detection model")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.person_det_model, args.ppe_detection_model)

import os
import xml.etree.ElementTree as ET
import argparse

def convert_voc_to_yolo(voc_dir, yolo_dir, classes_file):
    """Convert Pascal VOC XML annotations to YOLO format."""
    os.makedirs(yolo_dir, exist_ok=True)

    # Load class names
    with open(classes_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    for xml_file in os.listdir(voc_dir):
        if not xml_file.endswith(".xml"):
            continue

        tree = ET.parse(os.path.join(voc_dir, xml_file))
        root = tree.getroot()

        img_width = int(root.find("size/width").text)
        img_height = int(root.find("size/height").text)

        yolo_label_path = os.path.join(yolo_dir, xml_file.replace(".xml", ".txt"))

        with open(yolo_label_path, "w") as yolo_file:
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                if class_name not in classes:
                    continue  # Skip unknown classes

                class_id = classes.index(class_name)
                bbox = obj.find("bndbox")
                x_min = int(bbox.find("xmin").text)
                y_min = int(bbox.find("ymin").text)
                x_max = int(bbox.find("xmax").text)
                y_max = int(bbox.find("ymax").text)

                # Convert to YOLO format
                x_center = (x_min + x_max) / (2 * img_width)
                y_center = (y_min + y_max) / (2 * img_height)
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height

                yolo_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print(f"✅ Pascal VOC annotations converted and saved to {yolo_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PascalVOC XML to YOLO format.")
    parser.add_argument("input_dir", help="Path to the directory containing PascalVOC XML labels")
    parser.add_argument("output_dir", help="Path to save converted YOLO annotations")
    parser.add_argument("classes_file", help="Path to the classes.txt file")

    args = parser.parse_args()
    convert_voc_to_yolo(args.input_dir, args.output_dir, args.classes_file)

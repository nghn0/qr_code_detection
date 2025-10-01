import json
from pathlib import Path
from ultralytics import YOLO
import cv2

# Path to trained YOLO model
MODEL_PATH = "src/model/qr_yolo_model_aug/weights/best.pt"

# Folder containing images to run inference on
IMAGES_FOLDER = "QR_Dataset/test_images"  # or any folder with images

# Output folder for images with annotations
OUTPUT_IMAGE_FOLDER = "outputs/image_output"
Path(OUTPUT_IMAGE_FOLDER).mkdir(parents=True, exist_ok=True)

# Output JSON file
OUTPUT_JSON = "outputs/submission_detection_1.json"

# Load trained model
model = YOLO(MODEL_PATH)

# Collect all images
image_paths = list(Path(IMAGES_FOLDER).glob("*.jpg")) + list(Path(IMAGES_FOLDER).glob("*.png"))

results_json = []

for img_path in image_paths:
    # Run prediction
    results = model.predict(str(img_path), save=False, verbose=False)
    image_result = results[0]

    # Read original image for drawing
    img = cv2.imread(str(img_path))

    # Collect bounding boxes
    bboxes = []
    if image_result.boxes is not None:
        for box in image_result.boxes.xyxy.tolist():  # xyxy format
            x_min, y_min, x_max, y_max = [int(coord) for coord in box]
            bboxes.append({"bbox": [x_min, y_min, x_max, y_max]})

            # Draw rectangle on image
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    results_json.append({
        "image_id": img_path.stem,
        "qrs": bboxes
    })

    # Save annotated image
    output_img_path = Path(OUTPUT_IMAGE_FOLDER) / img_path.name
    cv2.imwrite(str(output_img_path), img)

# Save JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(results_json, f, indent=4)

print(f"✅ Inference complete! Annotated images saved in '{OUTPUT_IMAGE_FOLDER}' and JSON saved as '{OUTPUT_JSON}'")

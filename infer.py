import json
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import re

# ----------------------
# Paths
# ----------------------
MODEL_PATH = "src/model/qr_yolo_model_aug/weights/best.pt"
IMAGES_FOLDER = "QR_Dataset/test_images"
OUTPUT_IMAGE_FOLDER = "outputs/image_output"  # single folder
Path(OUTPUT_IMAGE_FOLDER).mkdir(parents=True, exist_ok=True)

OUTPUT_JSON_DET = "outputs/submission_detection_1.json"
OUTPUT_JSON_DEC = "outputs/submission_decoding_2.json"

# ----------------------
# Load YOLO model
# ----------------------
print("🔄 Loading YOLO model...")
model = YOLO(MODEL_PATH)


# ----------------------
# Optimized QR decoder
# ----------------------
def decode_qr_optimized(original_img, bbox):
    """
    Decode QR with preprocessing + padding
    """
    x1, y1, x2, y2 = map(int, bbox)
    pad = 30
    h, w = original_img.shape[:2]
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(w, x2 + pad)
    y2p = min(h, y2 + pad)

    crop = original_img[y1p:y2p, x1p:x2p]
    if crop.size == 0:
        return ""

    # Resize to minimum 400px
    min_size = 400
    if crop.shape[0] < min_size or crop.shape[1] < min_size:
        scale = max(min_size / crop.shape[0], min_size / crop.shape[1])
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    attempts = []

    # Preprocessing attempts
    attempts.append(gray)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    attempts.append(clahe.apply(gray))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    attempts.append(binary)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    attempts.append(adaptive)
    kernel = np.ones((2, 2), np.uint8)
    attempts.append(cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel))
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    attempts.append(cv2.filter2D(gray, -1, sharpen_kernel))

    qr_decoder = cv2.QRCodeDetector()
    for attempt in attempts:
        try:
            data, points, _ = qr_decoder.detectAndDecode(attempt)
            if data and len(data.strip()) > 0:
                return data.strip()
        except:
            continue
    return ""


# ----------------------
# Classification
# ----------------------
def classify_qr_type(value):
    """Classify QR content"""
    if not value:
        return "undecoded"
    v = value.upper()

    if any(kw in v for kw in ["BATCH", "LOT", "BATCH NO", "LOT NO", "BATCH#", "LOT#"]):
        return "batch"
    if v.startswith(("B", "L")) and len(value) >= 5 and any(c.isdigit() for c in value):
        return "batch"

    if any(kw in v for kw in ["EXP", "EXPIRY", "EXPIRE", "EXPY", "VALID UNTIL"]):
        return "expiry"
    if re.search(r'(20[2-9][0-9])(0[1-9]|1[0-2])', v):
        return "expiry"
    if re.search(r'(0[1-9]|1[0-2])/(20[2-9][0-9])', v):
        return "expiry"
    if re.search(r'\d{2}-\d{2}-\d{4}', v):
        return "expiry"

    if any(kw in v for kw in ["MRP", "PRICE", "RS", "₹", "RUPEES", "COST"]):
        return "price"
    if re.search(r'RS\s*\d+\.?\d*', v, re.IGNORECASE):
        return "price"

    if any(kw in v for kw in ["MFR", "MANUFACTURER", "MFG", "MADE BY", "SIG", "PRODUCED BY"]):
        return "manufacturer"

    if (8 <= len(value) <= 20 and value.isalnum()
            and any(c.isalpha() for c in value)
            and any(c.isdigit() for c in value)):
        return "serial"

    if (len(value) >= 6 and
            any(c.isupper() for c in value) and
            any(c.islower() for c in value) and
            any(c.isdigit() for c in value)):
        return "product_code"

    return "unknown"


# ----------------------
# Process images
# ----------------------
image_paths = list(Path(IMAGES_FOLDER).glob("*.jpg")) + list(Path(IMAGES_FOLDER).glob("*.png"))
results_det = []  # for submission_detection_1.json
results_dec = []  # for submission_decoding_2.json

total_detected, total_decoded = 0, 0
print(f"🔍 Found {len(image_paths)} test images.")

for img_path in image_paths:
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    preds = model.predict(str(img_path), save=False, verbose=False)
    boxes = preds[0].boxes.xyxy.cpu().numpy()

    bboxes_det = []  # for detection json
    bboxes_dec = []  # for decoding json

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        total_detected += 1
        bbox = [x1, y1, x2, y2]

        # Detection output (just bbox)
        bboxes_det.append({"bbox": bbox})

        # Decode with preprocessing + padding
        value = decode_qr_optimized(img, bbox)
        if value:
            total_decoded += 1
        qtype = classify_qr_type(value)

        bboxes_dec.append({
            "bbox": bbox,  # keep SAME bbox as detection
            "value": value,
            "type": qtype
        })

        # Draw annotation (infer.py style, no padding)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{value[:15]} ({qtype})", (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Add to JSON results
    results_det.append({"image_id": img_path.stem, "qrs": bboxes_det})
    results_dec.append({"image_id": img_path.stem, "qrs": bboxes_dec})

    # Save annotated image (single folder)
    cv2.imwrite(str(Path(OUTPUT_IMAGE_FOLDER) / img_path.name), img)

# ----------------------
# Save both JSONs
# ----------------------
with open(OUTPUT_JSON_DET, "w") as f:
    json.dump(results_det, f, indent=2)

with open(OUTPUT_JSON_DEC, "w") as f:
    json.dump(results_dec, f, indent=2)

success_rate = (total_decoded / total_detected * 100) if total_detected else 0
print(f"\n✅ Inference complete!")
print(f"📂 Annotated images in: {OUTPUT_IMAGE_FOLDER}")
print(f"📑 Detection JSON saved: {OUTPUT_JSON_DET}")
print(f"📑 Decoding JSON saved: {OUTPUT_JSON_DEC}")
print(f"📊 Total detected: {total_detected}, Decoded: {total_decoded}, Success rate: {success_rate:.1f}%")

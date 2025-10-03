


import os
import random
import shutil
from pathlib import Path


images_path = Path("QR_Dataset/train_images") 
labels_path = Path("labels") 
output_path = Path("QR_Dataset")  

train_ratio = 0.8


for split in ["train", "val"]:
    (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)


all_images = [f for f in images_path.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg"]]
random.shuffle(all_images)

train_size = int(len(all_images) * train_ratio)
train_files = all_images[:train_size]
val_files = all_images[train_size:]


def copy_files(files, split):
    for img_file in files:
        label_file = labels_path / (img_file.stem + ".txt")
        if label_file.exists():
            shutil.copy(img_file, output_path / "images" / split / img_file.name)
            shutil.copy(label_file, output_path / "labels" / split / label_file.name)
        else:
            print(f"⚠️ Label not found for {img_file.name}")



copy_files(train_files, "train")
copy_files(val_files, "val")


with open(output_path / "data.yaml", "w") as f:
    f.write(f"train: images/train\n")
    f.write(f"val: images/val\n")
    f.write("\n")
    f.write("nc: 1\n") 
    f.write("names: ['qr_code']\n")

print("✅ Dataset prepared and data.yaml created!")


from ultralytics import YOLO


model = YOLO("yolov8n.pt")  


results = model.train(
    data="QR_Dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="qr_yolo_model_aug",
    project="src/model",
    augment=True,
)

print("✅ Training finished! Model weights saved in src/model")

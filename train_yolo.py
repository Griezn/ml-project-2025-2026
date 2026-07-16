#!/usr/bin/env python3
import os
import sys
import shutil
import random
import json
import argparse
from datetime import datetime
import torch
from ultralytics import YOLO

def prepare_yolo_dataset(zombie_dataset_dir, yolo_dataset_dir, split_ratio=0.85, seed=42):
    """
    Converts absolute bounding box coordinates [x, y, w, h] to normalized YOLO format:
    [class_id, x_center, y_center, width, height]
    and splits the dataset into training and validation sets.
    """
    print("Preparing YOLO dataset...")
    random.seed(seed)
    
    images_src = os.path.join(zombie_dataset_dir, "images")
    labels_src = os.path.join(zombie_dataset_dir, "labels")
    
    if not os.path.exists(images_src) or not os.path.exists(labels_src):
        print(f"Error: Dataset source directories do not exist under '{zombie_dataset_dir}'")
        print("Please generate the dataset first using 'generate_zombie_data.py'")
        sys.exit(1)
        
    # Create YOLO folder structure
    for split in ["train", "val"]:
        os.makedirs(os.path.join(yolo_dataset_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(yolo_dataset_dir, "labels", split), exist_ok=True)
        
    image_files = [f for f in os.listdir(images_src) if f.endswith(".jpg")]
    if len(image_files) == 0:
        print(f"Error: No images found in '{images_src}'")
        sys.exit(1)
        
    print(f"Found {len(image_files)} images. Splitting into train/val...")
    
    random.shuffle(image_files)
    split_idx = int(len(image_files) * split_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Train samples: {len(train_files)}, Val samples: {len(val_files)}")
    
    # Image size in Knights Archers Zombies visual obs is 1280x720
    img_w, img_h = 1280, 720
    
    def process_split(files, split_name):
        for fname in files:
            # Copy image
            src_img_path = os.path.join(images_src, fname)
            dst_img_path = os.path.join(yolo_dataset_dir, "images", split_name, fname)
            shutil.copy(src_img_path, dst_img_path)
            
            # Convert label file
            label_name = fname.replace(".jpg", ".txt")
            src_lbl_path = os.path.join(labels_src, label_name)
            dst_lbl_path = os.path.join(yolo_dataset_dir, "labels", split_name, label_name)
            
            if os.path.exists(src_lbl_path):
                with open(src_lbl_path, "r") as f_in, open(dst_lbl_path, "w") as f_out:
                    for line in f_in:
                        parts = line.strip().split()
                        if len(parts) == 4:
                            x, y, w, h = map(float, parts)
                            # Convert to normalized YOLO format [0, x_center, y_center, w, h]
                            x_center = (x + w / 2.0) / img_w
                            y_center = (y + h / 2.0) / img_h
                            norm_w = w / img_w
                            norm_h = h / img_h
                            
                            # Clip values to [0.0, 1.0] just in case
                            x_center = max(0.0, min(1.0, x_center))
                            y_center = max(0.0, min(1.0, y_center))
                            norm_w = max(0.0, min(1.0, norm_w))
                            norm_h = max(0.0, min(1.0, norm_h))
                            
                            f_out.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
            else:
                # If label doesn't exist, create an empty file (meaning no objects/zombies)
                open(dst_lbl_path, "w").close()
                
    process_split(train_files, "train")
    process_split(val_files, "val")
    print("YOLO dataset preparation completed successfully.")

def write_dataset_yaml(yolo_dataset_dir, yaml_path):
    """Writes the dataset configuration YAML file required by YOLO."""
    abs_yolo_path = os.path.abspath(yolo_dataset_dir)
    yaml_content = f"""# Dataset config for Knights Archers Zombies (KAZ) zombie detection
path: {abs_yolo_path}
train: images/train
val: images/val

# Classes
names:
  0: zombie
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"Dataset YAML config written to {yaml_path}")

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv26 model on Zombie dataset")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, mps, cuda, or None for auto)")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--zombie_dir", type=str, default="zombie_dataset", help="Original zombie dataset directory")
    parser.add_argument("--yolo_dir", type=str, default="zombie_dataset_yolo", help="YOLO formatted dataset directory")
    parser.add_argument("--weights", type=str, default="yolo26n.pt", help="Pretrained weights file")
    
    args = parser.parse_args()
    
    # 1. Prepare YOLO dataset
    prepare_yolo_dataset(args.zombie_dir, args.yolo_dir)
    
    # 2. Write dataset configuration YAML
    yaml_path = os.path.join(args.yolo_dir, "zombie_dataset.yaml")
    write_dataset_yaml(args.yolo_dir, yaml_path)
    
    # 3. Determine device
    if args.device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Using training device: {device}")
    
    # 4. Generate distinctive run name and project directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"yolo26n_zombies_epochs{args.epochs}_lr{args.lr}_{timestamp}"
    project_dir = "yolo_training_runs"
    
    # 5. Load model
    print(f"Loading pretrained weights from {args.weights}...")
    if not os.path.exists(args.weights):
        print(f"Error: Weights file '{args.weights}' not found.")
        sys.exit(1)
        
    model = YOLO(args.weights)
    
    # 6. Save parameters for future comparison
    run_dir = os.path.join(project_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    params = {
        "weights": args.weights,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": device,
        "lr0": args.lr,
        "timestamp": timestamp,
        "classes": ["zombie"]
    }
    with open(os.path.join(run_dir, "parameters.json"), "w") as f:
        json.dump(params, f, indent=4)
    print(f"Saved run parameters to {os.path.join(run_dir, 'parameters.json')}")
    
    # 7. Start training
    print(f"Starting training run '{run_name}'...")
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        lr0=args.lr,
        project=project_dir,
        name=run_name,
        workers=2,
        val=True
    )
    
    # 8. Copy the best weights to project root for easy reference
    best_weights_path = os.path.join(run_dir, "weights", "best.pt")
    if os.path.exists(best_weights_path):
        target_weights_path = "best_yolo26n_zombie.pt"
        shutil.copy(best_weights_path, target_weights_path)
        print(f"Training completed successfully!")
        print(f"Best model weights saved to: '{target_weights_path}'")
    else:
        print("Warning: Could not find best.pt weights in the run directory.")

if __name__ == "__main__":
    main()

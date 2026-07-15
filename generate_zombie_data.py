#!/usr/bin/env python3
import os
import random
import argparse
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils import create_environment

def clip_bbox(x, y, w, h, img_w=1280, img_h=720):
    """
    Clips bounding box to image boundaries.
    Returns (x1, y1, w, h) or None if the box is completely off-screen.
    """
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    
    new_w = x2 - x1
    new_h = y2 - y1
    
    if new_w <= 2 or new_h <= 2:
        return None
    return int(x1), int(y1), int(new_w), int(new_h)

def generate_dataset(num_images=1000, output_dir="zombie_dataset", save_every=5):
    """
    Generates a dataset of distorted frames and zombie bounding boxes.
    """
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    meta_dir = os.path.join(output_dir, "metadata")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    
    print(f"Generating dataset at: {output_dir}")
    print(f"Target number of images: {num_images}")
    
    saved_count = 0
    pbar = tqdm(total=num_images)
    
    episode_id = 0
    
    while saved_count < num_images:
        # Choose a random distortion level
        dist_level = random.randint(0, 5)
        # Create environment
        env = create_environment(distortion_level=dist_level)
        env.reset()
        
        step_idx = 0
        cycle_idx = 0
        
        # Step through the environment
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                action = None
            else:
                # Sample random action
                action = env.action_space(agent).sample()
            
            env.step(action)
            
            # Since agents take turns, we process once per full cycle (when it is archer_0's turn)
            if agent == "archer_0" and not (termination or truncation):
                cycle_idx += 1
                
                # Subsample frames to reduce redundancy
                if cycle_idx % save_every == 0:
                    # Get the current full-screen visual observation
                    frame = env.observe("archer_0")
                    if frame is not None:
                        # Get active zombies
                        zombies = env.unwrapped.zombie_list
                        bboxes = []
                        
                        for zombie in zombies:
                            rx, ry = zombie.rect.x, zombie.rect.y
                            rw, rh = zombie.rect.width, zombie.rect.height
                            
                            clipped = clip_bbox(rx, ry, rw, rh)
                            if clipped is not None:
                                bboxes.append(clipped)
                        
                        # Only save if there are zombies on screen
                        if len(bboxes) > 0:
                            # Encode distortion level in filename
                            frame_name = f"frame_d{dist_level}_ep{episode_id:04d}_cy{cycle_idx:04d}"
                            
                            # Save image as JPEG
                            img_path = os.path.join(images_dir, f"{frame_name}.jpg")
                            img = Image.fromarray(frame)
                            img.save(img_path, "JPEG", quality=90)
                            
                            # Save labels (x y w h)
                            label_path = os.path.join(labels_dir, f"{frame_name}.txt")
                            with open(label_path, "w") as lf:
                                for bbox in bboxes:
                                    lf.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
                            
                            # Save metadata as JSON
                            meta_path = os.path.join(meta_dir, f"{frame_name}.json")
                            meta_data = {
                                "filename": f"{frame_name}.jpg",
                                "distortion_level": dist_level,
                                "episode_id": episode_id,
                                "cycle_idx": cycle_idx,
                                "num_zombies": len(bboxes)
                            }
                            with open(meta_path, "w") as mf:
                                json.dump(meta_data, mf, indent=4)
                            
                            saved_count += 1
                            pbar.update(1)
                            
                            if saved_count >= num_images:
                                break
            
            step_idx += 1
            
            # Limit episode length to prevent getting stuck in long runs
            if step_idx > 1000:
                break
                
        env.close()
        episode_id += 1
        
    pbar.close()
    print(f"Successfully generated {saved_count} images, label files, and metadata in '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate KAZ zombie detection dataset.")
    parser.add_argument("--num_images", type=int, default=1500, help="Number of images to generate")
    parser.add_argument("--output_dir", type=str, default="zombie_dataset", help="Output directory path")
    parser.add_argument("--save_every", type=int, default=5, help="Subsample rate of environment cycles")
    
    args = parser.parse_args()
    generate_dataset(num_images=args.num_images, output_dir=args.output_dir, save_every=args.save_every)

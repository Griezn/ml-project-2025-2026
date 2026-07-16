import os
import numpy as np
from typing import Callable
import gymnasium

class CustomWrapper:
    def __init__(self, env):
        self.env = env
    def __getattr__(self, name):
        return getattr(self.env, name)

class CustomPredictFunction:
    def __init__(self, env):
        pass
    def __call__(self, observation, agent, *args, **kwargs):
        return 0

class CustomZombieDetectorFunction(Callable):
    def __init__(self, env: gymnasium.Env):
        from ultralytics import YOLO
        package_directory = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(
            package_directory,
            "runs",
            "detect",
            "yolo_training_runs",
            "yolo26n_zombies_epochs30_lr0.01_20260715_205739",
            "weights",
            "best.pt"
        )
        if not os.path.exists(weights_path):
            weights_path = os.path.join(
                package_directory,
                "yolo_training_runs",
                "yolo26n_zombies_epochs30_lr0.01_20260715_205739",
                "weights",
                "best.pt"
            )
        self.model = YOLO(weights_path)

    def __call__(self, observation, *args, **kwargs):
        # Reshape flat observation (2764800,) to HWC (720, 1280, 3)
        img_arr = observation.reshape(720, 1280, 3)
        results = self.model(img_arr, verbose=False)
        
        zombie_rects = []
        if len(results) > 0:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = xyxy
                w = x2 - x1
                h = y2 - y1
                zombie_rects.append([x1, y1, w, h])
                
        return np.array(zombie_rects) if zombie_rects else np.zeros((0, 4))

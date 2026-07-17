import os
import sys
import random
from typing import Callable, Optional
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from ultralytics import YOLO

# Hyperparameters and dimensions
OBS_DIM = 45  # 5 (own) + 5 (teammate) + 35 (5 zombies * 7 features)
ACTION_DIM = 6

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, obs: torch.Tensor):
        logits = self.actor(obs)
        value = self.critic(obs)
        return logits, value


def shape_reward(agent_obs, action, raw_reward):
    """
    Shaped reward to guide exploration:
    - survival cost
    - reward shooting while facing a close zombie
    - penalty for shooting when no zombie is nearby or in sight
    - penalty if zombies get close to the bottom
    """
    shaped = float(raw_reward)
    
    # Parse zombie features:
    # 5 zombies (indices 10..44): rel_zx, rel_zy, norm_dist, cos_diff, sin_diff, zy / 720.0, is_detected
    closest_dist = 1.0
    closest_cos = -1.0
    closest_y = 0.0
    found_zombie = False
    
    for i in range(5):
        base_idx = 10 + i * 7
        is_detected = agent_obs[base_idx + 6]
        if is_detected > 0.5:
            dist = agent_obs[base_idx + 2]
            cos_diff = agent_obs[base_idx + 3]
            zy = agent_obs[base_idx + 5]
            if dist < closest_dist:
                closest_dist = dist
                closest_cos = cos_diff
                closest_y = zy
                found_zombie = True
                
    # If we shoot (action 4)
    if action == 4:
        if found_zombie:
            if closest_cos > 0.8:
                shaped += 0.1  # Reward for shooting while facing a zombie
            else:
                shaped -= 0.02 # Penalty for shooting in the wrong direction
        else:
            shaped -= 0.05 # Penalty for shooting with no zombies
            
    # Small survival penalty to encourage faster clears
    shaped -= 0.005
    
    # Penalty if zombie is getting close to the bottom line
    if found_zombie and closest_y > 0.85:
        shaped -= 0.05
        
    return shaped


class CustomWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # Load YOLO model for zombie detection
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
        
        # Initialize YOLO model on CPU/GPU
        self.yolo_model = YOLO(weights_path)
        
        # Cache for YOLO detections to prevent redundant runs for multiple agents at the same frame
        self._last_yolo_frame = -1
        self._last_yolo_detections = np.zeros((0, 4))
        
        # Flag to bypass YOLO during training to make it run 100x faster
        self.use_yolo = True
        self.shape_rewards = False

    def step(self, action):
        agent = self.env.agent_selection
        # Get agent observation before step modifies environment state
        obs = self.observe(agent)
        
        super().step(action)
        
        if self.shape_rewards and action is not None and agent in self.rewards:
            raw_reward = self.rewards[agent]
            shaped = shape_reward(obs, action, raw_reward)
            self.rewards[agent] = shaped

    def observation_space(self, agent: AgentID):
        # We return a 45-dimensional vector:
        # own_x, own_y, sin(angle), cos(angle), agent_id (5)
        # teammate_x, teammate_y, sin(angle), cos(angle), teammate_alive (5)
        # 5 closest zombies: rel_x, rel_y, dist, cos(diff), sin(diff), y/720, is_detected (35)
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

    def observe(self, agent: AgentID) -> ObsType | None:
        raw_obs = super().observe(agent)  # Shape (720, 1280, 3)
        if raw_obs is None:
            return np.zeros(OBS_DIM, dtype=np.float32)
            
        raw = self.unwrapped
        
        # Get frame count for caching
        current_frame = getattr(raw, "frames", 0)
        
        # Update cache if it is a new frame
        if self._last_yolo_frame != current_frame:
            if self.use_yolo:
                results = self.yolo_model(raw_obs, verbose=False)
                zombie_rects = []
                if len(results) > 0:
                    for box in results[0].boxes:
                        xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                        x1, y1, x2, y2 = xyxy
                        w = x2 - x1
                        h = y2 - y1
                        zombie_rects.append([x1, y1, w, h])
                self._last_yolo_detections = np.array(zombie_rects) if zombie_rects else np.zeros((0, 4))
            else:
                zombie_rects = []
                for z in raw.zombie_list:
                    zombie_rects.append([z.rect.x, z.rect.y, z.rect.width, z.rect.height])
                self._last_yolo_detections = np.array(zombie_rects) if zombie_rects else np.zeros((0, 4))
            self._last_yolo_frame = current_frame
            
        # Get own agent information
        archer_key = agent.replace("_", "")  # "archer0" or "archer1"
        if archer_key not in raw.archer_dict:
            # If the agent is dead/missing, return zeros
            return np.zeros(OBS_DIM, dtype=np.float32)
            
        own_archer = raw.archer_dict[archer_key]
        own_x = own_archer.rect.centerx
        own_y = own_archer.rect.centery
        own_angle_rad = np.radians(own_archer.angle)
        agent_id = 0.0 if archer_key == "archer0" else 1.0
        
        # Get teammate agent information
        teammate_key = "archer1" if archer_key == "archer0" else "archer0"
        if teammate_key in raw.archer_dict and raw.archer_dict[teammate_key].alive:
            teammate = raw.archer_dict[teammate_key]
            rel_team_x = (teammate.rect.centerx - own_x) / 1280.0
            rel_team_y = (teammate.rect.centery - own_y) / 720.0
            team_angle_rad = np.radians(teammate.angle)
            sin_team = np.sin(team_angle_rad)
            cos_team = np.cos(team_angle_rad)
            team_alive = 1.0
        else:
            rel_team_x = 0.0
            rel_team_y = 0.0
            sin_team = 0.0
            cos_team = 0.0
            team_alive = 0.0
            
        # Get zombie features (up to 5 closest)
        zombies_list = []
        for rect in self._last_yolo_detections:
            zx = rect[0] + rect[2] / 2.0
            zy = rect[1] + rect[3] / 2.0
            dist = np.sqrt((zx - own_x)**2 + (zy - own_y)**2)
            zombies_list.append((zx, zy, dist))
            
        # Sort by distance
        zombies_list.sort(key=lambda item: item[2])
        
        zombie_features = []
        for i in range(5):
            if i < len(zombies_list):
                zx, zy, dist = zombies_list[i]
                dx = zx - own_x
                dy = zy - own_y
                
                rel_zx = dx / 1280.0
                rel_zy = dy / 720.0
                norm_dist = dist / 1468.6
                
                if dist > 0:
                    ndx = dx / dist
                    ndy = dy / dist
                    # Archer heading direction
                    ax = np.sin(own_angle_rad)
                    ay = -np.cos(own_angle_rad)
                    cos_diff = ax * ndx + ay * ndy
                    sin_diff = ax * ndy - ay * ndx
                else:
                    cos_diff = 1.0
                    sin_diff = 0.0
                    
                zombie_features.extend([rel_zx, rel_zy, norm_dist, cos_diff, sin_diff, zy / 720.0, 1.0])
            else:
                zombie_features.extend([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
                
        # Combine everything
        obs_vector = np.array([
            own_x / 1280.0,
            own_y / 720.0,
            np.sin(own_angle_rad),
            np.cos(own_angle_rad),
            agent_id,
            rel_team_x,
            rel_team_y,
            sin_team,
            cos_team,
            team_alive
        ] + zombie_features, dtype=np.float32)
        
        return obs_vector


class CustomPredictFunction(Callable):
    def __init__(self, env):
        # Load weights from policy checkpoint
        package_directory = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(package_directory, "runs", "ippo_policy.pt")
        
        self.device = torch.device("cpu")
        self.policy = ActorCritic(obs_dim=OBS_DIM, action_dim=ACTION_DIM).to(self.device)
        
        if os.path.exists(weights_path):
            self.policy.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.policy.eval()
            print(f"Successfully loaded iPPO policy weights from {weights_path}")
        else:
            print(f"Warning: policy weights not found at {weights_path}. Running with random/untrained weights.")
            self.policy.eval()

    def __call__(self, observation, agent, *args, **kwargs):
        # Make a forward pass through the loaded policy
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.policy(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).item()
        return action


class CustomZombieDetectorFunction(Callable):
    def __init__(self, env):
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
        # Observation is HWC (720, 1280, 3)
        results = self.model(observation, verbose=False)
        zombie_rects = []
        if len(results) > 0:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = xyxy
                w = x2 - x1
                h = y2 - y1
                zombie_rects.append([x1, y1, w, h])
        return np.array(zombie_rects) if zombie_rects else np.zeros((0, 4))

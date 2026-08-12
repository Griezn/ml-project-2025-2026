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
        
        # Select best available hardware device (MPS for Apple Silicon GPU, CUDA, or CPU)
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Initialize YOLO model on selected device
        self.yolo_model = YOLO(weights_path)
        try:
            self.yolo_model.to(self.device)
        except Exception:
            pass
        
        # Cache for YOLO detections to prevent redundant runs for multiple agents at the same frame
        self._last_yolo_frame = -1
        self._last_yolo_detections = np.zeros((0, 4))
        
        # Flag to bypass YOLO during training to make it run 100x faster
        self.use_yolo = True
        self.shape_rewards = False

        # Locate VisualWrapper in wrapper stack and bypass heavy frame rendering/distortion during training
        curr = self.env
        while curr is not None:
            if hasattr(curr, "_refresh_transformed_frame"):
                orig_refresh = curr._refresh_transformed_frame
                wrapper_ref = self
                def fast_refresh(force=False, orig_ref=orig_refresh):
                    if wrapper_ref.use_yolo:
                        return orig_ref(force=force)
                    return
                curr._refresh_transformed_frame = fast_refresh
                break
            curr = getattr(curr, "env", None)

    def reset(self, seed=None, options=None):
        self._penalized_termination = False
        self._last_yolo_frame = -1
        self._last_yolo_detections = np.zeros((0, 4))
        self._prev_phi_dict = {}
        self._prev_phi_aim_dict = {}
        self.raw_episode_returns = {agent: 0.0 for agent in self.possible_agents}
        return super().reset(seed=seed, options=options)

    def step(self, action):
        agent = self.env.agent_selection
        obs = self.observe(agent)

        super().step(action)

        # Track raw reward before shaping
        if agent in self.rewards:
            self.raw_episode_returns[agent] = self.raw_episode_returns.get(agent, 0.0) + self.rewards[agent]

        if self.shape_rewards and action is not None and agent in self.rewards:
            self._apply_unified_reward_shaping(agent, action, obs)

        if self.terminations.get(agent, False) or self.truncations.get(agent, False):
            if agent in self.infos:
                self.infos[agent]["raw_return"] = self.raw_episode_returns.get(agent, 0.0)

    def _apply_unified_reward_shaping(self, agent, action, obs):
        """
        Potential-based shaping + real outcome rewards:
        - Kill reward: sparse, large, tied to an actual zombie being removed by a hit.
        - Distance-to-threat shaping: potential-based, so it only rewards *progress*,
          not just occupying a "bad" state.
        - Aim-progress shaping: potential-based alignment to the single nearest zombie.
        - Cooperative spacing: rewards *complementary coverage* instead of punishing proximity.
        - Shot economy: mild penalty only for shooting with nothing in range at all.
        - Global breach penalty.
        """
        raw = self.unwrapped
        gamma = 0.99  # match PPO discount factor

        # --- 1. Kill detection: compare zombie count before/after, excluding bottom-reach removals
        current_zombie_count = len(getattr(raw, "zombie_list", []))
        prev_zombie_count = getattr(self, "_prev_zombie_count", current_zombie_count)
        zombies_reached_bottom = getattr(self, "_zombies_reached_bottom_this_step", 0)

        zombies_killed = max(0, prev_zombie_count - current_zombie_count - zombies_reached_bottom)
        shaping_delta = zombies_killed * 3.0  # real, sparse, unambiguous signal

        self._prev_zombie_count = current_zombie_count

        # --- 2. Potential-based threat shaping
        # Φ(s) = -max distance-to-bottom over visible zombies (higher Φ = safer state)
        closest_zy = 0.0
        for i in range(5):
            base_idx = 10 + i * 7
            is_detected = obs[base_idx + 6]
            if is_detected > 0.5:
                zy_norm = obs[base_idx + 5]
                closest_zy = max(closest_zy, zy_norm)
        phi_now = -closest_zy

        prev_phi_dict = getattr(self, "_prev_phi_dict", {})
        phi_prev = prev_phi_dict.get(agent, phi_now)
        shaping_delta += gamma * phi_now - phi_prev
        prev_phi_dict[agent] = phi_now
        self._prev_phi_dict = prev_phi_dict

        # --- 3. Potential-based aim progress shaping (aimed at the single nearest zombie)
        nearest_cos_diff = 0.0
        best_dist = np.inf
        for i in range(5):
            base_idx = 10 + i * 7
            is_detected = obs[base_idx + 6]
            if is_detected > 0.5:
                norm_dist = obs[base_idx + 2]
                if norm_dist < best_dist:
                    best_dist = norm_dist
                    nearest_cos_diff = obs[base_idx + 3]

        phi_aim_now = nearest_cos_diff  # in [-1, 1], higher = better aligned to nearest threat
        prev_aim_dict = getattr(self, "_prev_phi_aim_dict", {})
        phi_aim_prev = prev_aim_dict.get(agent, phi_aim_now)
        shaping_delta += 0.3 * (gamma * phi_aim_now - phi_aim_prev)
        prev_aim_dict[agent] = phi_aim_now
        self._prev_phi_aim_dict = prev_aim_dict

        # --- 4. Shot economy: only penalize shooting with *nothing detected at all*
        if action == 4:
            any_zombie_detected = any(
                obs[10 + i * 7 + 6] > 0.5 for i in range(5)
            )
            if not any_zombie_detected:
                shaping_delta -= 0.02
        elif action != 4 and not any(obs[10 + i * 7 + 6] > 0.5 for i in range(5)):
            pass  # no zombies visible at all -- don't penalize scanning
        elif action != 4:
            shaping_delta -= 0.005  # tiny cost for disengaging while threats are visible

        # --- 5. Cooperative spacing: reward *coverage*, don't punish proximity outright
        rel_team_x = obs[5]
        rel_team_y = obs[6]
        team_alive = obs[9]
        if team_alive > 0.5 and closest_zy > 0.6:  # only matters when there's real pressure
            team_dist = np.sqrt(rel_team_x ** 2 + rel_team_y ** 2)
            if team_dist < 0.15:
                shaping_delta -= 0.02

        self.rewards[agent] += shaping_delta

        # --- 5. Global termination penalty (unchanged, still fine as a one-off)
        if not getattr(self, "_penalized_termination", False):
            if any(self.terminations.values()):
                self._penalized_termination = True
                for a in self.agents:
                    if a in self.rewards:
                        self.rewards[a] -= 2.0


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
        raw = self.unwrapped
        
        # Get frame count for caching
        current_frame = getattr(raw, "frames", 0)
        
        # Update cache if it is a new frame
        if self._last_yolo_frame != current_frame:
            if self.use_yolo:
                raw_obs = super().observe(agent)  # Shape (720, 1280, 3)
                if raw_obs is None:
                    return np.zeros(OBS_DIM, dtype=np.float32)
                # Run YOLO every 2 frames or if initial detection
                if self._last_yolo_frame == -1 or (current_frame - self._last_yolo_frame) >= 2 or len(self._last_yolo_detections) == 0:
                    import cv2
                    # Fast OpenCV resize to 640x360 before PyTorch tensor conversion
                    small_obs = cv2.resize(raw_obs, (640, 360))
                    with torch.no_grad():
                        results = self.yolo_model(small_obs, verbose=False, device=self.device, imgsz=320)
                    zombie_rects = []
                    if len(results) > 0 and results[0].boxes is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        for box in boxes:
                            # Scale bounding boxes back to 1280x720 coordinate space
                            x1, y1, x2, y2 = box[:4]
                            x1 *= 2.0
                            y1 *= 2.0
                            x2 *= 2.0
                            y2 *= 2.0
                            w = x2 - x1
                            h = y2 - y1
                            zombie_rects.append([x1, y1, w, h])
                    self._last_yolo_detections = np.array(zombie_rects) if zombie_rects else np.zeros((0, 4))
                    self._last_yolo_frame = current_frame
            else:
                zombie_rects = []
                for z in getattr(raw, "zombie_list", []):
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
        # Load MultiRLModule from the saved RLlib module checkpoint
        package_directory = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.join(package_directory, "runs", "ippo_rllib_module")
        
        self.device = torch.device("cpu")
        
        if os.path.exists(checkpoint_dir):
            from ray.rllib.core.rl_module import MultiRLModule
            try:
                self.modules = MultiRLModule.from_checkpoint(checkpoint_dir)
                if "shared_policy" in self.modules:
                    self.policy = self.modules["shared_policy"].to(self.device)
                else:
                    first_policy_id = list(self.modules.keys())[0]
                    self.policy = self.modules[first_policy_id].to(self.device)
                self.policy.eval()
                print(f"Successfully loaded RLlib policy from {checkpoint_dir}")
            except Exception as e:
                print(f"Error loading RLlib module checkpoint: {e}")
                self.policy = None
        else:
            print(f"Warning: RLlib module checkpoint not found at {checkpoint_dir}. Running with random/untrained weights.")
            self.policy = None

    def __call__(self, observation, agent, *args, **kwargs):
        if self.policy is None:
            # Fallback random action
            return random.randint(0, ACTION_DIM - 1)
            
        # Make a forward pass through the loaded RLlib policy module
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        fwd_ins = {"obs": obs_tensor}
        with torch.no_grad():
            fwd_outputs = self.policy.forward_inference(fwd_ins)
            action_dist_class = self.policy.get_inference_action_dist_cls()
            action_dist = action_dist_class.from_logits(fwd_outputs["action_dist_inputs"])
            action = action_dist.sample()[0].item()
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
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model = YOLO(weights_path)
        try:
            self.model.to(self.device)
        except Exception:
            pass

    def __call__(self, observation, *args, **kwargs):
        # Observation may arrive flat (2764800,) and needs reshaping to HWC (720, 1280, 3)
        img_arr = observation
        if isinstance(img_arr, np.ndarray) and img_arr.ndim == 1:
            img_arr = img_arr.reshape(720, 1280, 3)
        with torch.no_grad():
            results = self.model(img_arr, verbose=False, device=self.device, imgsz=416)
        zombie_rects = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box[:4]
                w = x2 - x1
                h = y2 - y1
                zombie_rects.append([x1, y1, w, h])
        return np.array(zombie_rects) if zombie_rects else np.zeros((0, 4))

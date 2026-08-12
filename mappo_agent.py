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

# Dimensions
OBS_DIM = 45  # 5 (own) + 5 (teammate) + 35 (5 zombies * 7 features)
STATE_DIM = 90  # Centralized state (concatenation of archer_0 and archer_1 obs)
ACTION_DIM = 6


class CustomWrapper(BaseWrapper):
    """
    MAPPO Custom Environment Wrapper:
    Wraps Knights Archers Zombies environment to extract structured 45-dim observations
    for decentralized policy execution and joint 90-dim state for centralized critic.
    """

    def __init__(self, env):
        super().__init__(env)

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

        self.yolo_model = YOLO(weights_path)
        try:
            self.yolo_model.to(self.device)
        except Exception:
            pass

        self._last_yolo_frame = -1
        self._last_yolo_detections = np.zeros((0, 4))

        self.use_yolo = True
        self.shape_rewards = False

        # Bypass heavy rendering during training
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

        if agent in self.rewards:
            self.raw_episode_returns[agent] = self.raw_episode_returns.get(agent, 0.0) + self.rewards[agent]

        if self.shape_rewards and action is not None and agent in self.rewards:
            self._apply_unified_reward_shaping(agent, action, obs)

        if self.terminations.get(agent, False) or self.truncations.get(agent, False):
            if agent in self.infos:
                self.infos[agent]["raw_return"] = self.raw_episode_returns.get(agent, 0.0)

    def _apply_unified_reward_shaping(self, agent, action, obs):
        raw = self.unwrapped
        gamma = 0.99

        current_zombie_count = len(getattr(raw, "zombie_list", []))
        prev_zombie_count = getattr(self, "_prev_zombie_count", current_zombie_count)
        zombies_reached_bottom = getattr(self, "_zombies_reached_bottom_this_step", 0)

        zombies_killed = max(0, prev_zombie_count - current_zombie_count - zombies_reached_bottom)
        shaping_delta = zombies_killed * 3.0

        self._prev_zombie_count = current_zombie_count

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

        phi_aim_now = nearest_cos_diff
        prev_aim_dict = getattr(self, "_prev_phi_aim_dict", {})
        phi_aim_prev = prev_aim_dict.get(agent, phi_aim_now)
        shaping_delta += 0.3 * (gamma * phi_aim_now - phi_aim_prev)
        prev_aim_dict[agent] = phi_aim_now
        self._prev_phi_aim_dict = prev_aim_dict

        if action == 4:
            any_zombie_detected = any(obs[10 + i * 7 + 6] > 0.5 for i in range(5))
            if not any_zombie_detected:
                shaping_delta -= 0.02
        elif action != 4 and not any(obs[10 + i * 7 + 6] > 0.5 for i in range(5)):
            pass
        elif action != 4:
            shaping_delta -= 0.005

        rel_team_x = obs[5]
        rel_team_y = obs[6]
        team_alive = obs[9]
        if team_alive > 0.5 and closest_zy > 0.6:
            team_dist = np.sqrt(rel_team_x ** 2 + rel_team_y ** 2)
            if team_dist < 0.15:
                shaping_delta -= 0.02

        self.rewards[agent] += shaping_delta

        if not getattr(self, "_penalized_termination", False):
            if any(self.terminations.values()):
                self._penalized_termination = True
                for a in self.agents:
                    if a in self.rewards:
                        self.rewards[a] -= 2.0

    def observation_space(self, agent: AgentID):
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

    def observe(self, agent: AgentID) -> ObsType | None:
        raw = self.unwrapped
        if agent not in raw.agents:
            return np.zeros(OBS_DIM, dtype=np.float32)

        archer = None
        for a in getattr(raw, "archer_list", []):
            if getattr(a, "agent_name", None) == agent:
                archer = a
                break

        if archer is None or not getattr(archer, "alive", True):
            return np.zeros(OBS_DIM, dtype=np.float32)

        own_x, own_y = archer.rect.centerx, archer.rect.centery
        own_angle_rad = np.radians(getattr(archer, "angle", 0.0))
        agent_id = 0.0 if agent == "archer_0" else 1.0

        teammate = None
        for a in getattr(raw, "archer_list", []):
            if getattr(a, "agent_name", None) != agent:
                teammate = a
                break

        if teammate is not None and getattr(teammate, "alive", True):
            tm_x, tm_y = teammate.rect.centerx, teammate.rect.centery
            tm_angle_rad = np.radians(getattr(teammate, "angle", 0.0))
            rel_team_x = (tm_x - own_x) / 1280.0
            rel_team_y = (tm_y - own_y) / 720.0
            sin_team = np.sin(tm_angle_rad)
            cos_team = np.cos(tm_angle_rad)
            team_alive = 1.0
        else:
            rel_team_x, rel_team_y, sin_team, cos_team, team_alive = 0.0, 0.0, 0.0, 0.0, 0.0

        if self.use_yolo:
            curr_step = getattr(raw, "num_moves", 0)
            if self._last_yolo_frame != curr_step:
                transformed_frame = getattr(raw, "transformed_frame", None)
                if transformed_frame is not None:
                    rgb_frame = np.transpose(transformed_frame, (1, 0, 2))
                    with torch.no_grad():
                        results = self.yolo_model(rgb_frame, verbose=False, device=self.device, imgsz=416)
                    zombie_rects = []
                    if len(results) > 0 and results[0].boxes is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        for box in boxes:
                            x1, y1, x2, y2 = box[:4]
                            w = x2 - x1
                            h = y2 - y1
                            zombie_rects.append([x1, y1, w, h])
                    self._last_yolo_detections = np.array(zombie_rects) if zombie_rects else np.zeros((0, 4))
                else:
                    self._last_yolo_detections = np.zeros((0, 4))
                self._last_yolo_frame = curr_step
            raw_zombies = self._last_yolo_detections
        else:
            raw_zombies = []
            for z in getattr(raw, "zombie_list", []):
                if getattr(z, "alive", True):
                    rx, ry = z.rect.x, z.rect.y
                    rw, rh = z.rect.width, z.rect.height
                    raw_zombies.append([rx, ry, rw, rh])
            raw_zombies = np.array(raw_zombies) if raw_zombies else np.zeros((0, 4))

        zombie_candidates = []
        for z_rect in raw_zombies:
            zx = z_rect[0] + z_rect[2] / 2.0
            zy = z_rect[1] + z_rect[3] / 2.0
            dx = zx - own_x
            dy = zy - own_y
            dist_sq = dx * dx + dy * dy
            zombie_candidates.append((dist_sq, zx, zy, dx, dy))

        zombie_candidates.sort(key=lambda item: item[0])
        top_5_zombies = zombie_candidates[:5]

        zombie_features = []
        for i in range(5):
            if i < len(top_5_zombies):
                dist_sq, zx, zy, dx, dy = top_5_zombies[i]
                dist = np.sqrt(dist_sq)
                rel_zx = dx / 1280.0
                rel_zy = dy / 720.0
                norm_dist = dist / 1468.6

                if dist > 0:
                    ndx = dx / dist
                    ndy = dy / dist
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
    """
    MAPPO Custom Predict Function:
    Loads trained MAPPO policy checkpoint. Evaluates decentralized policy (actor)
    given local observation during inference.
    """

    def __init__(self, env):
        package_directory = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.join(package_directory, "runs", "mappo_rllib_module")
        if not os.path.exists(checkpoint_dir):
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
                print(f"Successfully loaded MAPPO policy from {checkpoint_dir}")
            except Exception as e:
                print(f"Error loading MAPPO module checkpoint: {e}")
                self.policy = None
        else:
            print(f"Warning: MAPPO module checkpoint not found at {checkpoint_dir}. Running with random/untrained weights.")
            self.policy = None

    def __call__(self, observation, agent, *args, **kwargs):
        if self.policy is None:
            return random.randint(0, ACTION_DIM - 1)

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
        img = observation
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()

        if img.ndim == 1:
            img = img.reshape(720, 1280, 3)

        while len(img.shape) > 3:
            img = img[0]

        if len(img.shape) == 3 and img.shape[0] == 3 and img.shape[2] != 3:
            img = np.transpose(img, (1, 2, 0))

        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)

        img = np.ascontiguousarray(img)

        with torch.no_grad():
            results = self.model(img, verbose=False, device=self.device)
        zombie_rects = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box[:4]
                w = x2 - x1
                h = y2 - y1
                zombie_rects.append([x1, y1, w, h])
        return np.array(zombie_rects) if zombie_rects else np.zeros((0, 4))

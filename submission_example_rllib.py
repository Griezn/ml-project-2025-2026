"""
This file contains an example of implementation of the CustomWapper and CustomPredictFunction that you need to submit.

Here, we are using Ray RLLib to load the trained agents.
"""

from pathlib import Path
import random
from typing import Optional
from typing import Callable
import numpy as np
from PIL import Image
import torch
import gymnasium
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from ray.rllib.core.rl_module import MultiRLModule


class CustomWrapper(BaseWrapper):

    def __init__(self, env, target_size=(64, 64)):
        super().__init__(env)
        self.target_size = target_size  # (H, W)

    def observation_space(self, agent: AgentID):
        h, w = self.target_size
        c = super().observation_space(agent).shape[2]

        flat_size = h * w * c

        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(flat_size,),
            dtype=np.float32,
        )

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)  # (H, W, C) uint8

        # Resize
        img = Image.fromarray(obs)
        img = img.resize(self.target_size[::-1], Image.BILINEAR)
        obs_small = np.array(img)

        # Normalize + flatten
        flat_obs = obs_small.astype(np.float32) / 255.0
        return flat_obs.flatten()


class CustomPredictFunction(Callable):
    """ This is an example of an instantiation of the CustomPredictFunction that loads a trained RLLib algorithm from
    a checkpoint and extract the policies from it"""

    def __init__(self, env):

        # Here you should load your trained model(s) from a checkpoint in your folder
        best_checkpoint = (Path("results") / "learner_group" / "learner" / "rl_module").resolve()
        self.modules = MultiRLModule.from_checkpoint(best_checkpoint)

    def __call__(self, observation, agent, *args, **kwargs):
        rl_module = self.modules[agent]
        fwd_ins = {"obs": torch.Tensor(observation).unsqueeze(0)}
        fwd_outputs = rl_module.forward_inference(fwd_ins)
        action_dist_class = rl_module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(
            fwd_outputs["action_dist_inputs"]
        )
        action = action_dist.sample()[0].numpy()
        return action


class CustomZombieDetectorFunction(Callable):
    """Returns random detections."""

    def __init__(self, env: gymnasium.Env):
        pass

    def __call__(self, observation, *args, **kwargs):
        nb_zombies_detected = random.randint(0,4)
        zombie_rects = np.zeros((nb_zombies_detected, 4))
        for i in range(nb_zombies_detected):
            x = random.randint(0,1280-29)
            y = random.randint(0,720-31)
            w, h = 29, 31
            zombie_rects[i, :] = [x, y, w, h]
        return zombie_rects


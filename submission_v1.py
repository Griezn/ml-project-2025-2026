from pathlib import Path
from typing import Callable
import random

import pickle
import os
import gymnasium
import numpy as np
import torch
from PIL import Image
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from ray.rllib.core.rl_module import RLModuleSpec
from gymnasium import spaces
from kaz_impala_cnn_module import KAZImpalaCnnRLModule

IMG_SIZE = (84, 84)
NUM_FRAMES = 4


class CustomPredictFunction(Callable):
    def __init__(self, env):
        package_directory = os.path.dirname(os.path.abspath(__file__))

        obs_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3 * NUM_FRAMES, *IMG_SIZE),
            dtype=np.float32,
        )
        act_space = env.action_space(env.possible_agents[0])

        spec = RLModuleSpec(
            module_class=KAZImpalaCnnRLModule,
            observation_space=obs_space,
            action_space=act_space,
            model_config={"hidden_dim": 256, "channels": [16, 32, 32]},
            inference_only=True,
        )
        self.module = spec.build()

        weights_path = os.path.join(
            package_directory,
            "results_best",
            "learner_group",
            "learner",
            "rl_module",
            "shared_policy",
            "module_state.pkl",
        )

        with open(weights_path, "rb") as f:
            saved_state = pickle.load(f)

        self.module.set_state(saved_state)
        self.module.eval()

    def __call__(self, observation, agent, *args, **kwargs):
        fwd_ins = {"obs": torch.Tensor(observation).unsqueeze(0)}
        fwd_outputs = self.module.forward_inference(fwd_ins)
        action_dist_class = self.module.get_inference_action_dist_cls()
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
        nb_zombies_detected = random.randint(0, 4)
        zombie_rects = np.zeros((nb_zombies_detected, 4))
        for i in range(nb_zombies_detected):
            x = random.randint(0, 1280 - 29)
            y = random.randint(0, 720 - 31)
            w, h = 29, 31
            zombie_rects[i, :] = [x, y, w, h]
        return zombie_rects

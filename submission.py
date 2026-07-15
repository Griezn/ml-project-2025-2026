from typing import Callable
import random

import pickle
import os
import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType
from ray.rllib.core.columns import Columns
from gymnasium import spaces
from typing import Any, Dict, Optional
from ray.rllib.core.rl_module import RLModule

IMG_SIZE = (84, 84)
NUM_FRAMES = 4


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv0(out)
        out = F.relu(out)
        out = self.conv1(out)
        return out + x


class ConvSequence(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(out_channels)
        self.res_block1 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x


class KAZImpalaCnnRLModule(TorchRLModule, ValueFunctionAPI):

    @override(TorchRLModule)
    def setup(self):
        c, h, w = self.observation_space.shape
        num_actions = self.action_space.n

        hidden_dim = self.model_config.get("hidden_dim", 256)
        channels = self.model_config.get("channels", [16, 32, 32])

        conv_seqs = []
        in_ch = c
        cur_h, cur_w = h, w
        for out_ch in channels:
            conv_seqs.append(ConvSequence(in_ch, out_ch))
            in_ch = out_ch
            cur_h = (cur_h + 1) // 2
            cur_w = (cur_w + 1) // 2
        self.conv_seqs = nn.ModuleList(conv_seqs)

        flat_size = in_ch * cur_h * cur_w

        self.fc = nn.Linear(flat_size, hidden_dim)
        nn.init.orthogonal_(self.fc.weight)

        self.policy_head = nn.Linear(hidden_dim, num_actions)
        nn.init.orthogonal_(self.policy_head.weight)

        if not self.inference_only:
            self.value_head = nn.Linear(hidden_dim, 1)
            nn.init.orthogonal_(self.value_head.weight)

    def _encode(self, obs):
        x = obs.float()
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = F.relu(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

    @override(TorchRLModule)
    def _forward_inference(self, batch, **kwargs):
        with torch.no_grad():
            features = self._encode(batch[Columns.OBS])
            logits = self.policy_head(features)
        return {Columns.ACTION_DIST_INPUTS: logits}

    @override(TorchRLModule)
    def _forward_exploration(self, batch, **kwargs):
        with torch.no_grad():
            features = self._encode(batch[Columns.OBS])
            logits = self.policy_head(features)
        return {Columns.ACTION_DIST_INPUTS: logits}

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        features = self._encode(batch[Columns.OBS])
        logits = self.policy_head(features)
        return {Columns.ACTION_DIST_INPUTS: logits, Columns.EMBEDDINGS: features,}

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None,) -> TensorType:
        if embeddings is None:
            embeddings = self._encode(batch[Columns.OBS])
        return self.value_head(embeddings).squeeze(-1)



class CustomWrapper(BaseWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.target_size = IMG_SIZE
        self.num_frames = NUM_FRAMES
        self._frame_buffers = {}

    def observation_space(self, agent: AgentID):
        h, w = self.target_size
        c = 3  # RGB
        return spaces.Box(low=0.0, high=1.0, shape=(c * self.num_frames, h, w), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self._frame_buffers.clear()
        return super().reset(seed=seed, options=options)

    def _process_frame(self, obs):
        img = Image.fromarray(obs)
        img = img.resize(self.target_size[::-1], Image.BILINEAR)
        obs_float = np.array(img).astype(np.float32) / 255.0
        return np.transpose(obs_float, (2, 0, 1))  # (H, W, 3) -> (3, H, W)

    def _get_stacked_obs(self, agent, frame):
        if agent not in self._frame_buffers:
            self._frame_buffers[agent] = [frame.copy() for _ in range(self.num_frames)]
        self._frame_buffers[agent].append(frame)
        self._frame_buffers[agent] = self._frame_buffers[agent][-self.num_frames:]
        return np.concatenate(self._frame_buffers[agent], axis=0)

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        if obs is None:
            return None
        frame = self._process_frame(obs)
        return self._get_stacked_obs(agent, frame)



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

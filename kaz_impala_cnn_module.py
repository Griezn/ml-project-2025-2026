"""
IMPALA CNN RLModule

3 ConvSequences with channels [16, 32, 32], each containing:
  conv3x3 -> maxpool -> 2x ResidualBlock

From Espeholt et al. 2018 (IMPALA paper)

Template from tiny atari example from rllib github:
https://github.com/ray-project/ray/blob/master/rllib/examples/rl_modules/classes/tiny_atari_cnn_rlm.py
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType


class ResidualBlock(nn.Module):
    # Pre-activation residual block: ReLU -> Conv -> ReLU -> Conv + skip

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
    # Conv3x3 -> MaxPool(3,stride=2) -> ResBlock -> ResBlock

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
    """
    (12, 84, 84) with 4-frame stacking (rgb=3)

    - 3 ConvSequences with residual blocks (15 conv layers total)
    - Separate policy head and value head
    - Shared encoder

    """

    @override(TorchRLModule)
    def setup(self):
        c, h, w = self.observation_space.shape
        num_actions = self.action_space.n

        hidden_dim = self.model_config.get("hidden_dim", 256)
        channels = self.model_config.get("channels", [16, 32, 32])

        # IMPALA CNN encoder
        conv_seqs = []
        in_ch = c
        cur_h, cur_w = h, w
        for out_ch in channels:
            conv_seqs.append(ConvSequence(in_ch, out_ch))
            in_ch = out_ch
            cur_h = (cur_h + 1) // 2  # maxpool stride 2 with padding 1
            cur_w = (cur_w + 1) // 2
        self.conv_seqs = nn.ModuleList(conv_seqs)

        flat_size = in_ch * cur_h * cur_w

        # ICLR PPO blog says orthogonal is better for early stability in training
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
        x = x.reshape(x.size(0), -1)  # flatten instead of GAP
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
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.EMBEDDINGS: features,
        }

    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> TensorType:
        if embeddings is None:
            embeddings = self._encode(batch[Columns.OBS])
        return self.value_head(embeddings).squeeze(-1)

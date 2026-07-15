"""
Nature DQN CNN RLModule

From Mnih, V. et al. 2015 Nature 518

Template from tiny atari example from rllib github:
https://github.com/ray-project/ray/blob/master/rllib/examples/rl_modules/classes/tiny_atari_cnn_rlm.py
"""



from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType


class KAZCnnRLModule(TorchRLModule, ValueFunctionAPI):
    """

    (12, 84, 84) with 4-frame stacking (rgb=3)
    
    - Nature DQN-style 3-layer CNN encoder
    - Separate policy head and value head
    - Shared encoder

    """

    @override(TorchRLModule)
    def setup(self):
        c, h, w = self.observation_space.shape 
        num_actions = self.action_space.n       

        hidden_dim = self.model_config.get("hidden_dim", 256) 
        conv_filters = self.model_config.get("conv_filters")

        if conv_filters is None:
            # Default: Nature DQN-style (same as comp ling AI colab but 2D)
            conv_filters = [
                [32, 5, 2],  # [num_filters, kernel_size, stride]
                [64, 3, 2],  # no padding used as said in the Nature DQN paer Mnih et al., 2015
                [64, 3, 1],  # because we need to compress image down to small feature vector
            ]                # this is standard for visual RL encoders, because spatial dim need to shrink at each layer

        # CNN encoder from conv_filters config
        layers = []
        in_channels = c
        for out_channels, kernel_size, stride in conv_filters:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.Flatten()) # need to flatten for compact vector for policy/value heads
        self.encoder = nn.Sequential(*layers)

        # Compute the flattened feature size after the conv layers
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            flat_size = self.encoder(dummy).shape[1] # for diff img sizes

        
        self.policy_head = nn.Sequential(
            nn.Linear(flat_size, hidden_dim),  # flat features -> hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),  # hidden -> one logit per action
        )

       
        if not self.inference_only:
            self.value_head = nn.Sequential(
                nn.Linear(flat_size, hidden_dim),  # flat features -> hidden layer
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),  # hidden -> single value estimate V(s)
            )

    @override(TorchRLModule)
    def _forward_inference(self, batch, **kwargs):
        with torch.no_grad():
            obs = batch[Columns.OBS].float()
            features = self.encoder(obs)
            logits = self.policy_head(features)
        return {Columns.ACTION_DIST_INPUTS: logits}

    @override(TorchRLModule)
    def _forward_exploration(self, batch, **kwargs):
        with torch.no_grad():
            obs = batch[Columns.OBS].float()
            features = self.encoder(obs)
            logits = self.policy_head(features)
        return {Columns.ACTION_DIST_INPUTS: logits}

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        obs = batch[Columns.OBS].float()
        features = self.encoder(obs)  
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
            obs = batch[Columns.OBS].float()
            embeddings = self.encoder(obs)
        # squeeze(-1) turns shape (B, 1) into (B,) -> PPO expects a 1D tensor
        return self.value_head(embeddings).squeeze(-1)
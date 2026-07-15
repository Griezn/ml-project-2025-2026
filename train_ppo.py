#!/usr/bin/env python3

from pathlib import Path
import random
import os
import shutil

import numpy as np
import torch
import pettingzoo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from utils import create_environment

from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from PIL import Image
from kaz_cnn_module import *
from visual_utils import set_distortion_level

IMG_SIZE = (84, 84)  # 84x84: Atari standard
NUM_FRAMES = 4       # Number of frames to stack

# 1. PRE-PROCESS

class TrainingWrapper(BaseWrapper):
    """
    Preprocesses pixel observations for CNN input.

    grayscale -> resize -> normalize to [0,1] -> (C,H,W) transpose -> frame stack (for CNN)

    Frame stacking: stacks the last N frames along the channel dimension so the
    agent can perceive motion (which direction a zombie is moving).
    With 1 RGB channel (grayscale) and num_frames=4, the observation becomes (4, H, W).
    """

    def __init__(self, env, target_size=IMG_SIZE, num_frames=NUM_FRAMES,
                 randomize_distortion=True, reward_shaping=False):
        super().__init__(env)
        self.target_size = target_size  # (H, W)
        self.num_frames = num_frames
        self.randomize_distortion = randomize_distortion
        self.reward_shaping = reward_shaping
        # stores last N processed frames
        self._frame_buffers = {}
        self._step_count = 0
        self._episode_done = False

    def observation_space(self, agent: AgentID):
        h, w = self.target_size
        c = 3  # RGB
        return spaces.Box(low=0.0, high=1.0, shape=(c * self.num_frames, h, w), dtype=np.float32)

    def _process_frame(self, obs):
        # resize, normalize, and return (1, H, W) float32
        img = Image.fromarray(obs)
        img = img.resize(self.target_size[::-1], Image.BILINEAR)
        obs_float = np.array(img).astype(np.float32) / 255.0
        return np.transpose(obs_float, (2, 0, 1))  # (H, W, 12) -> (12, H, W)

    def _get_stacked_obs(self, agent, frame):
        # Add frame to buffer and return stacked observation
        if agent not in self._frame_buffers:
            self._frame_buffers[agent] = [frame.copy() for _ in range(self.num_frames)]

        self._frame_buffers[agent].append(frame)
        self._frame_buffers[agent] = self._frame_buffers[agent][-self.num_frames:]

        return np.concatenate(self._frame_buffers[agent], axis=0)

    # Need to change reset to be able to do randomizing on zombies+dist and for frame stacking clearance
    def reset(self, seed=None, options=None):
        self._frame_buffers.clear()
        self._step_count = 0
        self._episode_done = False
        if self.randomize_distortion:
            level = random.randint(0,5)
            set_distortion_level(level)
        result = super().reset(seed=seed, options=options)
        # Randomize max_zombies AFTER reset (KAZ reset restores original value)
        try:
            base_env = self.env.env.env.env.aec_env.env.env.env
            base_env.max_zombies = random.randint(3, 5)
        except AttributeError:
            pass
        return result

    def step(self, action):
        out = super().step(action)
        self._step_count += 1
        # Step with optional reward shaping
        if self.reward_shaping:
            self._apply_reward_shaping()

        return out

    def _apply_reward_shaping(self):
        """
        1. Death penalty: -1.0 - opposite of killing zombie
        2. Arrow near-zombie bonus: +0.02 max — gradient toward accurate shooting
        3. Zombie danger penalty: -0.005 * danger — gentle nudge to intercept zombies early
        4. Zombie facing bonus: +0.015 * facing * prox_fact - learn agents to face zombies to shoot
        """
        
        try:
            base_env = self.env.env.env.env.aec_env.env.env.env
            zombie_list = list(base_env.zombie_list)
        except AttributeError:
            return 

        # Death penalty
        for agent in self.agents:
            if agent in self.rewards:
                if self.terminations.get(agent, False) and not self._episode_done:
                    self.rewards[agent] -= 1

        if any(self.terminations.get(a, False) for a in self.agents):
            self._episode_done = True

        # Zombie danger penalty (for all agents)
        # 40% lower half of the screen is danger
        DANGER_Y = 430
        MAX_Y = 715
        for zombie in zombie_list:
            zy = zombie.rect.centery
            if zy > DANGER_Y:
                danger = (zy - DANGER_Y) / (MAX_Y - DANGER_Y) # Procent of how far
                penalty = -0.005 * danger
                for agent in self.agents:
                    if agent in self.rewards:
                        self.rewards[agent] += penalty


    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        if obs is None:
            return None
        
        frame = self._process_frame(obs)
        if self.num_frames > 1:
            return self._get_stacked_obs(agent, frame)
        return frame


# 2. ALGORITHM CONFIG

def make_ppo_config(env_id, max_iterations=500):

    module_class = KAZCnnRLModule
    model_config = {
        "hidden_dim": 512,
        "conv_filters": [
            [32, 8, 4],    # Nature DQN architecture for 84x84
            [64, 4, 2],
            [64, 3, 1],
        ],
    }

    rl_module_specs = {
        "shared_policy": RLModuleSpec(
            module_class=module_class,
            model_config=model_config,
        )
    }
    policy_mapping_fn = lambda agent_id, *args, **kwargs: "shared_policy"
    actual_policies = {"shared_policy"}
    actual_policies_to_train = {"shared_policy"}

    # LR schedule: linear decay
    total_timesteps = max_iterations * 4096
    lr_schedule = [
        [0, 2.5e-4],
        [total_timesteps, 1e-5],
    ]

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(env=env_id, disable_env_checking=True)
        .env_runners(
            num_env_runners=4,
            sample_timeout_s=300.0,
            rollout_fragment_length="auto",
        )
        .learners(
            num_learners=2,
            num_gpus_per_learner=1,
        )
        .multi_agent(
            policies=actual_policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=actual_policies_to_train,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(rl_module_specs=rl_module_specs)
        )
        .training( # S marked are from the https://openreview.net/pdf?id=WoLQsYU8aZ
            use_critic=True,
            use_gae=True,
            lr=lr_schedule,
            gamma=0.998,
            lambda_=0.95, # S
            clip_param=0.1, # S
            entropy_coeff=0.01, # S
            vf_loss_coeff=0.5, # Shared policy, so shared encoder so 0.5
            train_batch_size=4096,
            grad_clip=0.5, # Atari PPO standard for pixel
            num_epochs=3,
            minibatch_size=512,
        )
        .debugging(log_level="WARN")
    )

    return config


# 3. TRAINING

def train(max_iterations=500, checkpoint_path="results", reward_shaping=True):

    env_id = "kaz_training"
 
    def env_creator(config):
        base_env = create_environment(distortion_level=0)
        wrapped = TrainingWrapper(base_env, randomize_distortion=True, reward_shaping=reward_shaping)
        parallel_env = pettingzoo.utils.conversions.aec_to_parallel(wrapped)
        return ParallelPettingZooEnv(parallel_env)

    register_env(env_id, env_creator)

    # Seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Build config
    temp_env = create_environment(distortion_level=0)
    temp_env.reset()
    temp_env.close()
    config = make_ppo_config(env_id, max_iterations=max_iterations)

    algo = config.build()
    checkpoint_dir = str(Path(checkpoint_path).resolve())
    best_checkpoint_dir = str(Path(checkpoint_path + "_best").resolve())

    best_reward = -float("inf")
    reward_history = []

    WINDOW = 20
    PATIENCE = 80

    best_rolling_avg = -float("inf")
    iters_since_improvement = 0

    for i in range(max_iterations):
        result = algo.train()
        result.pop("config")

        if "env_runners" in result and "agent_episode_returns_mean" in result["env_runners"]:
            mean_rewards = result["env_runners"]["agent_episode_returns_mean"]
            total_mean = sum(mean_rewards.values()) / len(mean_rewards)
            reward_history.append(total_mean)
            print(f"Iter {i:4d} | Mean reward: {total_mean:.2f} | Per agent: {mean_rewards}")

            if total_mean > best_reward:
                best_reward = total_mean
                print(f"  >> New best: {best_reward:.2f} — saving best checkpoint")
                save_result = algo.save(best_checkpoint_dir)
                print(f"     Saved to: {save_result.checkpoint.path}")

            if len(reward_history) >= WINDOW:
                rolling_avg = np.mean(reward_history[-WINDOW:])
                if rolling_avg > best_rolling_avg + 0.1:
                    best_rolling_avg = rolling_avg
                    iters_since_improvement = 0
                else:
                    iters_since_improvement += 1

                if iters_since_improvement >= PATIENCE:
                    print(f"\n  Early stopping, rolling avg hasn't improved for {PATIENCE} iters")
                    print(f"  Best rolling avg: {best_rolling_avg:.2f}, Best single: {best_reward:.2f}")
                    break

        if i % 20 == 0:
            save_result = algo.save(checkpoint_dir)
            print(f"  Checkpoint saved: {save_result.checkpoint.path}")

    # Final save
    save_result = algo.save(checkpoint_dir)

    algo.stop()
    return reward_history



if __name__ == "__main__":

    train(max_iterations=500, checkpoint_path="results", reward_shaping=True)

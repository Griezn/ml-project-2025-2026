import os
import sys
import json
import time
import csv
from datetime import datetime
import numpy as np
import torch

import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from pettingzoo.utils.conversions import aec_to_parallel

from utils import create_environment
from ippo_agent import CustomWrapper, OBS_DIM, ACTION_DIM

# Training hyperparameters (matching the manual implementation)
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
EPOCHS = 8
BATCH_SIZE = 64
ENT_COEF = 0.01
VAL_COEF = 0.5
MAX_GRAD_NORM = 0.5
ROLLOUT_STEPS = 2048  # Total environment steps per policy update


def convert_rllib_to_pytorch(rllib_state_dict):
    """
    Converts DefaultPPOTorchRLModule state dict keys into the format expected 
    by our lightweight ActorCritic model class in ippo_agent.py
    """
    pytorch_state_dict = {}
    
    # Actor mapping
    pytorch_state_dict["actor.0.weight"] = rllib_state_dict["encoder.actor_encoder.net.mlp.0.weight"]
    pytorch_state_dict["actor.0.bias"] = rllib_state_dict["encoder.actor_encoder.net.mlp.0.bias"]
    pytorch_state_dict["actor.2.weight"] = rllib_state_dict["encoder.actor_encoder.net.mlp.2.weight"]
    pytorch_state_dict["actor.2.bias"] = rllib_state_dict["encoder.actor_encoder.net.mlp.2.bias"]
    pytorch_state_dict["actor.4.weight"] = rllib_state_dict["pi.net.mlp.0.weight"]
    pytorch_state_dict["actor.4.bias"] = rllib_state_dict["pi.net.mlp.0.bias"]
    
    # Critic mapping
    pytorch_state_dict["critic.0.weight"] = rllib_state_dict["encoder.critic_encoder.net.mlp.0.weight"]
    pytorch_state_dict["critic.0.bias"] = rllib_state_dict["encoder.critic_encoder.net.mlp.0.bias"]
    pytorch_state_dict["critic.2.weight"] = rllib_state_dict["encoder.critic_encoder.net.mlp.2.weight"]
    pytorch_state_dict["critic.2.bias"] = rllib_state_dict["encoder.critic_encoder.net.mlp.2.bias"]
    pytorch_state_dict["critic.4.weight"] = rllib_state_dict["vf.net.mlp.0.weight"]
    pytorch_state_dict["critic.4.bias"] = rllib_state_dict["vf.net.mlp.0.bias"]
    
    return pytorch_state_dict


def env_creator(config):
    """Constructor for the environment registered with Ray."""
    base_env = create_environment(distortion_level=0)
    wrapped_env = CustomWrapper(base_env)
    wrapped_env.use_yolo = False  # Bypasses YOLO during training for a 100x speedup!
    wrapped_env.shape_rewards = True  # Enable shaped rewards during training!
    parallel_env = aec_to_parallel(wrapped_env)
    return ParallelPettingZooEnv(parallel_env)


def train():
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ippo_rllib_run_{timestamp}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    # Save hyperparameters
    params = {
        "lr": LR,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_eps": CLIP_EPS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "ent_coef": ENT_COEF,
        "val_coef": VAL_COEF,
        "max_grad_norm": MAX_GRAD_NORM,
        "rollout_steps": ROLLOUT_STEPS,
        "framework": "rllib"
    }
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=4)
        
    # Setup metrics logging
    csv_file = open(os.path.join(run_dir, "metrics.csv"), "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "update", "episodes", "steps", "mean_raw_return", 
        "mean_shaped_return", "actor_loss", "critic_loss", 
        "entropy", "fps"
    ])
    csv_file.flush()

    # Register the environment creator
    register_env("knights_archers_zombies_v10_rllib", env_creator)
    
    # Configure RLlib PPO
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(env="knights_archers_zombies_v10_rllib", disable_env_checking=True)
        .env_runners(
            num_env_runners=1,
            rollout_fragment_length="auto"
        )
        .multi_agent(
            # Standard single policy shared across all agents (Independent PPO / parameter sharing)
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            policies_to_train={"shared_policy"},
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "shared_policy": RLModuleSpec(
                        module_class=PPOTorchRLModule,
                        model_config={"fcnet_hiddens": [128, 128], "fcnet_activation": "tanh"}
                    )
                }
            )
        )
        .training(
            train_batch_size=ROLLOUT_STEPS,
            minibatch_size=BATCH_SIZE,
            num_epochs=EPOCHS,
            lr=LR,
            gamma=GAMMA,
            lambda_=GAE_LAMBDA,
            clip_param=CLIP_EPS,
            entropy_coeff=ENT_COEF,
            vf_loss_coeff=VAL_COEF,
            grad_clip=MAX_GRAD_NORM
        )
        .debugging(log_level="ERROR")
    )

    # Build the algorithm
    print("Building RLlib PPO Algorithm...")
    algo = config.build()
    
    max_updates = 2
    print(f"Starting training for {max_updates} updates...")
    
    start_time = time.time()
    
    for update_idx in range(1, max_updates + 1):
        result = algo.train()
        
        # Extract metrics
        env_runners_stats = result.get("env_runners", {})
        learners_stats = result.get("learners", {}).get("shared_policy", {})
        
        episodes = env_runners_stats.get("num_episodes_lifetime", 0)
        steps = result.get("num_env_steps_sampled_lifetime", 0)
        mean_shaped_return = env_runners_stats.get("episode_return_mean", 0.0)
        
        # Since we modify rewards inside CustomWrapper.step, the RLlib environment runner return matches the shaped return.
        # We report this as both raw and shaped returns in the CSV.
        mean_raw_return = mean_shaped_return
        
        actor_loss = learners_stats.get("policy_loss", 0.0)
        critic_loss = learners_stats.get("vf_loss", 0.0)
        entropy = learners_stats.get("entropy", 0.0)
        
        # Compute local training throughput
        time_elapsed = time.time() - start_time
        fps = int(result.get("num_env_steps_sampled_this_iter", ROLLOUT_STEPS) / result.get("time_this_iter_s", 1.0))
        
        print(f"Update {update_idx}/{max_updates} | Steps: {steps} | Episodes: {episodes} | "
              f"Raw Return: {mean_raw_return:.2f} | Shaped Return: {mean_shaped_return:.2f} | "
              f"Loss Actor/Critic: {actor_loss:.4f}/{critic_loss:.4f} | Ent: {entropy:.3f} | FPS: {fps}")
              
        # Log to CSV
        csv_writer.writerow([
            update_idx, episodes, steps, mean_raw_return,
            mean_shaped_return, actor_loss, critic_loss,
            entropy, fps
        ])
        csv_file.flush()
        
        # Save checkpoints and export standard PyTorch state dict
        if update_idx == 1 or update_idx % 5 == 0:
            # 1. Save standard RLlib checkpoint
            algo.save(run_dir)
            
            # 2. Extract weights from RLlib and export to standard PyTorch .pt file
            try:
                rl_module = algo.get_module("shared_policy")
                rllib_sd = rl_module.state_dict()
                pytorch_sd = convert_rllib_to_pytorch(rllib_sd)
                
                # Save to run-specific checkpoint path
                torch.save(pytorch_sd, os.path.join(run_dir, f"checkpoint_{update_idx}.pt"))
                torch.save(pytorch_sd, os.path.join(run_dir, "best_policy.pt"))
                
                # Overwrite standard workspace location for CustomPredictFunction
                torch.save(pytorch_sd, "runs/ippo_policy.pt")
                print(f"Exported PyTorch policy weights to runs/ippo_policy.pt")
            except Exception as e:
                print(f"Warning: Failed to export PyTorch weights: {e}")
                
    csv_file.close()
    algo.stop()
    ray.shutdown()
    print("Training finished successfully!")


if __name__ == "__main__":
    train()

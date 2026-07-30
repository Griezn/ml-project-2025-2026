import os
import sys
import json
import time
import csv
import argparse
from datetime import datetime
import numpy as np
import torch

import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.callbacks.callbacks import RLlibCallback
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


class CustomMetricsCallback(RLlibCallback):
    def on_episode_end(self, *, episode, env_runner, metrics_logger, **kwargs):
        if metrics_logger is not None and hasattr(episode, "get_infos"):
            raw_returns = []
            for agent_id, agent_infos in episode.get_infos().items():
                for step_info in agent_infos:
                    if isinstance(step_info, dict) and "raw_return" in step_info:
                        raw_returns.append(step_info["raw_return"])
            if raw_returns:
                metrics_logger.log_value("raw_episode_return", sum(raw_returns) / len(raw_returns))


def env_creator(config):
    """Constructor for the environment registered with Ray."""
    base_env = create_environment(distortion_level=0)
    wrapped_env = CustomWrapper(base_env)
    wrapped_env.use_yolo = False  # Bypasses YOLO during training!
    wrapped_env.shape_rewards = True  # Enable shaped rewards during training!
    parallel_env = aec_to_parallel(wrapped_env)
    return ParallelPettingZooEnv(parallel_env)


def find_latest_checkpoint(runs_dir="runs"):
    if not os.path.exists(runs_dir):
        return None
    candidates = []
    for d in os.listdir(runs_dir):
        full_path = os.path.join(runs_dir, d)
        if os.path.isdir(full_path):
            if os.path.exists(os.path.join(full_path, "algorithm_state.pkl")) or \
               os.path.exists(os.path.join(full_path, "learner_group")) or \
               os.path.exists(os.path.join(full_path, "rllib_checkpoint.json")):
                candidates.append((full_path, os.path.getmtime(full_path)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def parse_args():
    parser = argparse.ArgumentParser(description="Train IPPO Agent with RLlib")
    parser.add_argument("--max-updates", type=int, default=100, help="Number of training updates for fresh training or additional updates for restored run")
    parser.add_argument("--run-name", type=str, default=None, help="Distinctive run name")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--rollout-steps", type=int, default=ROLLOUT_STEPS, help="Rollout steps per update")
    parser.add_argument("--ent-coef", type=float, default=0.05, help="Entropy coefficient for exploration")
    parser.add_argument("--restore-checkpoint", type=str, default=None, help="Path to checkpoint directory to restore from, or 'latest' / 'auto'")
    return parser.parse_args()


def train(args=None):
    if args is None:
        args = parse_args()

    ent_coef = getattr(args, "ent_coef", ENT_COEF)

    # Register the environment creator
    register_env("knights_archers_zombies_v10_rllib", env_creator)

    # Determine restore path if specified
    restore_path = None
    if args.restore_checkpoint:
        if args.restore_checkpoint.lower() in ["latest", "auto", "true"]:
            restore_path = find_latest_checkpoint()
            if restore_path is None:
                print("Warning: No checkpoint found to restore! Starting fresh training.")
        else:
            restore_path = args.restore_checkpoint
        if restore_path and not os.path.exists(restore_path):
            print(f"Error: Specified checkpoint path '{restore_path}' does not exist.")
            sys.exit(1)

    # Setup run directories under a distinctive name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_name = args.run_name
    elif restore_path:
        restored_name = os.path.basename(os.path.normpath(restore_path))
        run_name = f"ippo_cont_from_{restored_name}_{timestamp}"
    else:
        run_name = f"ippo_rllib_run_{timestamp}"

    run_dir = os.path.abspath(os.path.join("runs", run_name))
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    # Check previous update count from restored metrics if available
    start_update = 0
    if restore_path:
        prev_metrics_file = os.path.join(restore_path, "metrics.csv")
        if os.path.exists(prev_metrics_file):
            try:
                with open(prev_metrics_file, "r") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    last_row = None
                    for row in reader:
                        if row:
                            last_row = row
                    if last_row:
                        start_update = int(float(last_row[0]))
            except Exception as e:
                print(f"Could not read previous update count: {e}")

    # Save hyperparameters and run metadata
    params = {
        "lr": args.lr,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_eps": CLIP_EPS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "ent_coef": ent_coef,
        "val_coef": VAL_COEF,
        "max_grad_norm": MAX_GRAD_NORM,
        "rollout_steps": args.rollout_steps,
        "max_updates": args.max_updates,
        "start_update": start_update,
        "target_end_update": start_update + args.max_updates,
        "restored_from": os.path.abspath(restore_path) if restore_path else None,
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

    if restore_path:
        abs_ckpt = os.path.abspath(restore_path)
        print(f"Restoring RLlib PPO Algorithm from checkpoint: {abs_ckpt}...")
        algo = Algorithm.from_checkpoint(abs_ckpt)
        print(f"Successfully restored algorithm! Continuing from update {start_update} for {args.max_updates} more updates.")
    else:
        # Configure RLlib PPO
        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .environment(env="knights_archers_zombies_v10_rllib", disable_env_checking=True)
            .callbacks(CustomMetricsCallback)
            .env_runners(
                num_env_runners=6,
                rollout_fragment_length="auto"
            )
            .multi_agent(
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
                train_batch_size=args.rollout_steps,
                minibatch_size=BATCH_SIZE,
                num_epochs=EPOCHS,
                lr=args.lr,
                gamma=GAMMA,
                lambda_=GAE_LAMBDA,
                clip_param=CLIP_EPS,
                entropy_coeff=ent_coef,
                vf_loss_coeff=VAL_COEF,
                grad_clip=MAX_GRAD_NORM
            )
            .debugging(log_level="ERROR")
        )
        print("Building RLlib PPO Algorithm...")
        algo = config.build()

    max_updates = args.max_updates
    print(f"Starting training for {max_updates} updates (run: {run_name})...")
    
    start_time = time.time()
    
    for i in range(1, max_updates + 1):
        global_update_idx = start_update + i
        result = algo.train()
        
        # Extract metrics
        env_runners_stats = result.get("env_runners", {})
        learners_stats = result.get("learners", {}).get("shared_policy", {})
        
        episodes = env_runners_stats.get("num_episodes_lifetime", 0)
        steps = result.get("num_env_steps_sampled_lifetime", 0)
        mean_shaped_return = env_runners_stats.get("episode_return_mean", 0.0)
        mean_raw_return = env_runners_stats.get("raw_episode_return", mean_shaped_return)
        
        actor_loss = learners_stats.get("policy_loss", 0.0)
        critic_loss = learners_stats.get("vf_loss", 0.0)
        entropy = learners_stats.get("entropy", 0.0)
        
        # Compute local training throughput
        fps = int(result.get("num_env_steps_sampled_this_iter", ROLLOUT_STEPS) / result.get("time_this_iter_s", 1.0))
        
        print(f"Update {global_update_idx} ({i}/{max_updates}) | Steps: {steps} | Episodes: {episodes} | "
              f"Raw Return: {mean_raw_return:.2f} | Shaped Return: {mean_shaped_return:.2f} | "
              f"Loss Actor/Critic: {actor_loss:.4f}/{critic_loss:.4f} | Ent: {entropy:.3f} | FPS: {fps}")
              
        # Log to CSV
        csv_writer.writerow([
            global_update_idx, episodes, steps, mean_raw_return,
            mean_shaped_return, actor_loss, critic_loss,
            entropy, fps
        ])
        csv_file.flush()
        
        # Save checkpoints and export the rl_module natively
        if i == 1 or i % 5 == 0 or i == max_updates:
            # 1. Save standard RLlib checkpoint
            save_result = algo.save(run_dir)
            checkpoint_path = save_result.checkpoint.path
            print(f"Saved RLlib checkpoint to {checkpoint_path}")
            
            # 2. Export the RLlib MultiRLModule natively via save_to_path
            try:
                import shutil
                dest_rl_module = os.path.abspath("runs/ippo_rllib_module")
                if os.path.exists(dest_rl_module):
                    shutil.rmtree(dest_rl_module)
                
                algo.env_runner.module.save_to_path(dest_rl_module)
                print(f"Exported RLlib module natively to {dest_rl_module}")
                
                # Also save a copy inside the run directory for history comparison
                history_module_dir = os.path.abspath(os.path.join(run_dir, "ippo_rllib_module"))
                if os.path.exists(history_module_dir):
                    shutil.rmtree(history_module_dir)
                algo.env_runner.module.save_to_path(history_module_dir)
            except Exception as e:
                print(f"Warning: Failed to export RLlib module: {e}")
                
    csv_file.close()
    algo.stop()
    ray.shutdown()
    print("Training finished successfully!")


if __name__ == "__main__":
    train()

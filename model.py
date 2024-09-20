# model.py

import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)

from environment import create_env, create_eval_env

LOG_DIR = os.getenv("LOG_DIR", "./logs")


def create_model(params):
    env_id = os.getenv("ENV_NAME")
    n_envs = params.get("n_envs", 1)

    env = create_env(env_id, n_envs)
    eval_env = create_eval_env(env_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO(
        "MlpPolicy", env, verbose=1, device=device, tensorboard_log=LOG_DIR, **params
    )

    return model, env, eval_env


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    try:
        model = PPO.load(model_path)
        print(f"Model loaded from {model_path}")

        env_id = os.getenv("ENV_NAME")

        # Try to get n_envs from the model, fall back to environment variable if not available
        try:
            n_envs = model.n_envs
        except AttributeError:
            n_envs = int(os.getenv("N_ENVS", 1))  # Default to 1 if N_ENVS is not set

        # Use the create_env function
        env = create_env(env_id, n_envs)

        # Use the create_eval_env function
        eval_env = create_eval_env(env_id)

        # Set the environment for the model
        model.set_env(env)

        return model, env, eval_env
    except Exception as e:
        raise ValueError(f"Error loading model: {str(e)}")


def train_model(model, env, eval_env, params):
    total_timesteps = int(os.getenv("TOTAL_TIMESTEPS"))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.getenv("MODEL_DIR"),
        log_path=os.path.join(LOG_DIR, "eval_logs"),
        eval_freq=int(os.getenv("EVAL_FREQ", 10000)),
        deterministic=True,
        render=False,
        n_eval_episodes=int(os.getenv("N_EVAL_EPISODES", 5)),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=int(os.getenv("CHECKPOINT_FREQ", 100000)),
        save_path=os.path.join(os.getenv("MODEL_DIR"), "checkpoints"),
        name_prefix="ppo_model",
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")

    return model, eval_callback.best_mean_reward

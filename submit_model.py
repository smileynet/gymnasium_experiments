import gymnasium as gym
import os
from dotenv import load_dotenv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder
from huggingface_sb3 import package_to_hub


def dry_run(model, env_id, n_eval_episodes=10, video_length=1000):
    # Create the evaluation environment
    eval_env = make_vec_env(env_id, n_envs=1)

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=n_eval_episodes
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Create a video of the model's performance
    video_env = VecVideoRecorder(
        eval_env,
        "videos",
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix=f"{env_id}-preview",
    )

    obs = video_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = video_env.step(action)
        if done:
            obs = video_env.reset()

    video_env.close()
    print(f"Preview video saved in 'videos' directory")


# Load environment variables
load_dotenv()

# Get the username from .env file
username = os.getenv("HF_USERNAME")
if not username:
    raise ValueError("HF_USERNAME not set in .env file")

# Create the environment
env_id = "LunarLander-v2"
env = make_vec_env(env_id, n_envs=1)

# Load the best model
model_path = os.path.join("models", "best_model.zip")
model = PPO.load(model_path, env=env)

# Define model architecture
model_architecture = "PPO"

# Create repo ID
repo_id = f"{username}/{model_architecture}-{env_id}"

# Perform dry run
dry_run(model, env_id)

# Ask user if they want to submit to Hugging Face Hub
submit = input("Do you want to submit the model to Hugging Face Hub? (y/n): ").lower()

if submit == "y":
    # This method save, evaluate, generate a model card and record a replay video of your agent before pushing the repo to the hub
    package_to_hub(
        model=model,
        model_name=f"{model_architecture}-{env_id}",
        model_architecture=model_architecture,
        env_id=env_id,
        eval_env=env,
        repo_id=repo_id,
        commit_message="Submitting best model",
    )
    print("Model submitted to Hugging Face Hub")
else:
    print("Model submission cancelled")

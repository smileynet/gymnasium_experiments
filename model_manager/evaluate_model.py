import json
import os
from datetime import datetime
import sys

import gymnasium as gym
import moviepy.config as mpconfig
from dotenv import load_dotenv
from sqlalchemy import Null
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from logging_config import console, logger

load_dotenv()


def evaluate_and_record(
    model, env_id, video_name, n_eval_episodes=10, video_length=1000
):
    """
    Evaluate a model and record a video of its performance.

    Args:
        model: The trained model to evaluate.
        env_id (str): The ID of the environment to evaluate on.
        video_name (str): Custom name for the video file.
        n_eval_episodes (int): Number of episodes to evaluate.
        video_length (int): Length of the recorded video.

    Returns:
        dict: Evaluation results including mean reward, standard deviation, and metadata.

    Raises:
        Exception: If an error occurs during evaluation or recording.
    """
    try:

        def make_env():
            env = gym.make(env_id, render_mode="rgb_array")
            return Monitor(env)

        logger.debug(f"Loading eval environment for {env_id}")
        eval_env = DummyVecEnv([make_env for _ in range(1)])
        logger.debug(f"Observation space: {eval_env.observation_space}")
        logger.debug(f"Action space: {eval_env.action_space}")

        outputs_dir = os.getenv("OUTPUTS_DIR", "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        video_dir = os.path.join(outputs_dir, "videos")

        eval_env = VecVideoRecorder(
            eval_env,
            video_dir,
            record_video_trigger=lambda x: x == 0,
            video_length=video_length,
            name_prefix=video_name,
        )

        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=n_eval_episodes
        )

        eval_env.close()

        console.print(
            f"[green]Preview video '[bold]{video_name}[/bold]' saved in '[bold]{video_dir}[/bold]' directory[/green]"
        )
        console.print(
            f"[cyan]Mean reward: [bold]{mean_reward:.2f}[/bold] +/- [bold]{std_reward:.2f}[/bold][/cyan]"
        )

        results = {
            "env_id": env_id,
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "n_eval_episodes": n_eval_episodes,
            "eval_datetime": datetime.now().isoformat(),
        }

        results_path = os.path.join(outputs_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        console.print(
            f"[green]Evaluation results saved to [bold]{results_path}[/bold][/green]"
        )

        return results
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}")

    try:
        env_id = os.getenv("ENV_NAME", "LunarLander-v2")
        model_dir = os.getenv("MODEL_DIR", "models")
        best_model_name = os.getenv("BEST_MODEL_NAME", "best_model.zip")
        n_eval_episodes = int(os.getenv("N_EVAL_EPISODES", 10))
        video_name = os.getenv("STUDY_NAME", "model_evaluation")

        model_path = os.path.join(model_dir, best_model_name)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading model from {model_path}")
        logger.info(f"Model loaded from {model_path}")
        evaluate_and_record(model, env_id, video_name, n_eval_episodes=n_eval_episodes)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

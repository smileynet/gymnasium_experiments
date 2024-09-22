import json
import logging
import os
from datetime import datetime

import gymnasium as gym
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
            return gym.make(env_id, render_mode="rgb_array")

        logging.debug(f"Loading eval environment for {env_id}")
        eval_env = DummyVecEnv([make_env for _ in range(1)])
        logging.debug(f"Observation space: {eval_env.observation_space}")
        logging.debug(f"Action space: {eval_env.action_space}")

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

        logging.info(f"Preview video '{video_name}' saved in '{video_dir}' directory")
        logging.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

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

        logging.info(f"Evaluation results saved to {results_path}")

        return results
    except Exception as e:
        logging.error(f"An error occurred during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        env_id = os.getenv("ENV_NAME", "LunarLander-v2")
        model_dir = os.getenv("MODEL_DIR", "models")
        best_model_name = os.getenv("BEST_MODEL_NAME", "best_model.zip")
        n_eval_episodes = int(os.getenv("N_EVAL_EPISODES", 10))
        video_name = os.getenv("STUDY_NAME", "model_evaluation")

        model_path = os.path.join(model_dir, best_model_name)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logging.info(f"Loading model from {model_path}")
        model = PPO.load(model_path)
        logging.info(f"Model loaded from {model_path}")

        evaluate_and_record(model, env_id, video_name, n_eval_episodes=n_eval_episodes)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

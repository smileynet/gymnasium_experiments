import os
import logging
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_and_record(model, env_id, n_eval_episodes=10, video_length=1000, output_dir="outputs"):
    """
    Evaluate a model and record a video of its performance.

    Args:
        model: The trained model to evaluate.
        env_id (str): The ID of the environment to evaluate on.
        n_eval_episodes (int): Number of episodes to evaluate.
        video_length (int): Length of the recorded video.
        output_dir (str): Directory to save outputs.

    Returns:
        tuple: Mean reward and standard deviation of the evaluation.
    """
    try:
        eval_env = make_vec_env(env_id, n_envs=1)

        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=n_eval_episodes
        )


        os.makedirs(output_dir, exist_ok=True)

        video_env = VecVideoRecorder(
            eval_env,
            os.path.join(output_dir, "videos"),
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

        logging.info(f"Preview video saved in '{output_dir}/videos' directory")
        logging.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        return mean_reward, std_reward
    except Exception as e:
        logging.error(f"An error occurred during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        load_dotenv()

        env_id = os.getenv("ENV_NAME", "LunarLander-v2")
        model_path = os.path.join("models", "best_model.zip")
        output_dir = os.getenv("OUTPUTS_DIR", "outputs")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logging.info(f"Loading model from {model_path}")
        model = PPO.load(model_path)
        logging.info(f"Model loaded from {model_path}")

        mean_reward, std_reward = evaluate_and_record(model, env_id, output_dir=output_dir)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

import os
import json
import argparse
import logging
import torch
from dotenv import load_dotenv
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from torch.utils.tensorboard import SummaryWriter

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_model(params):
    try:
        # Create directories for logs and models
        os.makedirs("./logs", exist_ok=True)
        os.makedirs(os.getenv('MODEL_DIR'), exist_ok=True)

        # Set up TensorBoard writer
        writer = SummaryWriter(log_dir="./logs")

        # Create and wrap the environment
        env = gym.make(os.getenv('ENV_NAME'))
        env = Monitor(env, "./logs")
        env = DummyVecEnv([lambda: env])

        # Create eval environment
        eval_env = gym.make(os.getenv('ENV_NAME'))
        eval_env = Monitor(eval_env, "./logs")
        eval_env = DummyVecEnv([lambda: eval_env])

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize the agent with given parameters
        model = PPO("MlpPolicy", env, verbose=1, device=device, tensorboard_log="./logs", **params)

        # Configure logger
        new_logger = configure("./logs", ["tensorboard", "stdout"])
        model.set_logger(new_logger)

        # Create EvalCallback with early stopping
        eval_callback = EvalCallback(
            eval_env,
            n_eval_episodes=int(os.getenv('N_EVAL_EPISODES', 10)),
            eval_freq=int(os.getenv('EVAL_FREQ', 100000)),
            log_path="./logs/",
            best_model_save_path=os.getenv('MODEL_DIR'),
            callback_on_new_best=None,
            callback_after_eval=StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=int(os.getenv('EARLY_STOPPING_PATIENCE', 3)),
                min_evals=5,
                verbose=1
            ),
            verbose=1,
        )

        best_mean_reward = float("-inf")
        for epoch in range(int(os.getenv('N_EPOCHS', 10))):
            logger.info(f"Starting epoch {epoch + 1}/{os.getenv('N_EPOCHS', 10)}")

            # Train the agent
            model.learn(
                total_timesteps=int(os.getenv('TOTAL_TIMESTEPS')),
                callback=eval_callback,
                reset_num_timesteps=False,
                tb_log_name="PPO",
            )

            # Evaluate the model
            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=int(os.getenv('N_EVAL_EPISODES', 10)))
            logger.info(f"Epoch {epoch + 1} - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

            # Log metrics to TensorBoard
            writer.add_scalar("eval/mean_reward", mean_reward, epoch)
            writer.add_scalar("eval/std_reward", std_reward, epoch)

            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                save_path = os.path.join(os.getenv('MODEL_DIR'), os.getenv('BEST_MODEL_NAME'))
                model.save(save_path)
                logger.info(f"New best model saved to {save_path}")
            else:
                logger.info("No improvement in mean reward. Stopping training.")
                break

        writer.close()
        return model, best_mean_reward

    except Exception as e:
        logger.exception(f"An error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL model with best parameters.")
    parser.add_argument("--params", type=str, required=True, help="JSON string of model parameters")

    args = parser.parse_args()

    try:
        params = json.loads(args.params)
    except json.JSONDecodeError:
        logger.error("Invalid JSON string for parameters")
        exit(1)

    try:
        model, best_mean_reward = train_model(params)
        logger.info(f"Training completed. Best mean reward: {best_mean_reward:.2f}")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        exit(1)

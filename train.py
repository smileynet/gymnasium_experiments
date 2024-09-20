# python train.py --params '{"learning_rate": 0.001, "n_steps": 2048, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.01}' --env_name LunarLander-v2 --total_timesteps 1000000 --save_path best_model.zip --n_epochs 10 --early_stopping_patience 3

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import os
import json
import argparse
import logging
import torch
from torch.utils.tensorboard import SummaryWriter

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_model(
    params,
    env_name,
    total_timesteps,
    save_path,
    n_eval_episodes=10,
    eval_freq=100000,
    n_epochs=10,
    early_stopping_patience=3,
):
    try:
        # Create directories for logs and models
        os.makedirs("./logs", exist_ok=True)
        os.makedirs("./models", exist_ok=True)

        # Set up TensorBoard writer
        writer = SummaryWriter(log_dir="./logs")

        # Create and wrap the environment
        env = gym.make(env_name)
        env = Monitor(env, "./logs")
        env = DummyVecEnv([lambda: env])

        # Create eval environment
        eval_env = gym.make(env_name)
        eval_env = Monitor(eval_env, "./logs")
        eval_env = DummyVecEnv([lambda: eval_env])

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize the agent with given parameters
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            tensorboard_log="./logs",
            **params,
        )

        # Configure logger
        new_logger = configure("./logs", ["tensorboard", "stdout"])
        model.set_logger(new_logger)

        # Create EvalCallback with early stopping
        eval_callback = EvalCallback(
            eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path="./logs/",
            best_model_save_path="./best_model/",
            callback_on_new_best=None,
            callback_after_eval=StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=early_stopping_patience, min_evals=5, verbose=1
            ),
            verbose=1,
        )

        best_mean_reward = float("-inf")
        for epoch in range(n_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{n_epochs}")

            # Train the agent
            model.learn(
                total_timesteps=total_timesteps,
                callback=eval_callback,
                reset_num_timesteps=False,
                tb_log_name="PPO",
            )

            # Evaluate the model
            mean_reward, std_reward = evaluate_policy(
                model, eval_env, n_eval_episodes=n_eval_episodes
            )
            logger.info(
                f"Epoch {epoch + 1} - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}"
            )

            # Log metrics to TensorBoard
            writer.add_scalar("eval/mean_reward", mean_reward, epoch)
            writer.add_scalar("eval/std_reward", std_reward, epoch)

            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
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
    parser = argparse.ArgumentParser(
        description="Train an RL model with best parameters."
    )
    parser.add_argument(
        "--params", type=str, required=True, help="JSON string of model parameters"
    )
    parser.add_argument(
        "--env_name", type=str, required=True, help="Name of the Gym environment"
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        required=True,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save the trained model"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Number of evaluations with no improvement after which training will be stopped",
    )

    args = parser.parse_args()

    try:
        params = json.loads(args.params)
    except json.JSONDecodeError:
        logger.error("Invalid JSON string for parameters")
        exit(1)

    try:
        model, best_mean_reward = train_model(
            params,
            args.env_name,
            args.total_timesteps,
            args.save_path,
            n_epochs=args.n_epochs,
            early_stopping_patience=args.early_stopping_patience,
        )
        logger.info(f"Training completed. Best mean reward: {best_mean_reward:.2f}")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        exit(1)

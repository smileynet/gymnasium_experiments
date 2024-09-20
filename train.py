import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

def train_model(params, env_name, total_timesteps, save_path=None, n_eval_episodes=10, eval_freq=10000, n_epochs=10, early_stopping_patience=3):
    # Create the environment
    env = gym.make(env_name)
    env = DummyVecEnv([lambda: env])

    # Create eval environment
    eval_env = gym.make(env_name)
    eval_env = DummyVecEnv([lambda: eval_env])

    # Initialize the agent with given parameters
    model = PPO("MlpPolicy", env, verbose=1, **params)

    # Create EvalCallback with early stopping
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=n_eval_episodes,
        eval_freq=eval_freq,
        log_path="./logs/",
        best_model_save_path="./best_model/",
        callback_on_new_best=None,
        callback_after_eval=StopTrainingOnNoModelImprovement(max_no_improvement_evals=early_stopping_patience, min_evals=5, verbose=1),
        verbose=1
    )

    best_mean_reward = float('-inf')
    for epoch in range(n_epochs):
        print(f"Starting epoch {epoch + 1}/{n_epochs}")

        # Train the agent
        model.learn(total_timesteps=total_timesteps, callback=eval_callback, reset_num_timesteps=False)

        # Evaluate the model
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes)
        print(f"Epoch {epoch + 1} - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            if save_path:
                model.save(save_path)
                print(f"New best model saved to {save_path}")
        else:
            print("No improvement in mean reward. Stopping training.")
            break

    return model, best_mean_reward

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Train an RL model.")
    parser.add_argument("--params", type=str, required=True, help="JSON string of model parameters")
    parser.add_argument("--env_name", type=str, required=True, help="Name of the Gym environment")
    parser.add_argument("--total_timesteps", type=int, required=True, help="Total timesteps for training")
    parser.add_argument("--save_path", type=str, help="Path to save the trained model")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Number of evaluations with no improvement after which training will be stopped")

    args = parser.parse_args()

    params = json.loads(args.params)

    model, best_mean_reward = train_model(
        params,
        args.env_name,
        args.total_timesteps,
        args.save_path,
        n_epochs=args.n_epochs,
        early_stopping_patience=args.early_stopping_patience
    )

    print(f"Training completed. Best mean reward: {best_mean_reward:.2f}")

# python train.py --params '{"learning_rate": 0.001}' --env_name LunarLander-v2 --total_timesteps 1000000 --save_path my_model
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv


def experiment_model(params, env_name, total_timesteps, n_eval_episodes=10):
    # Create the environment
    env = gym.make(env_name)
    env = DummyVecEnv([lambda: env])

    # Create eval environment
    eval_env = gym.make(env_name)
    eval_env = DummyVecEnv([lambda: eval_env])

    # Initialize the agent with given parameters
    model = PPO("MlpPolicy", env, verbose=1, **params)

    # Train the agent
    model.learn(total_timesteps=total_timesteps)

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=n_eval_episodes
    )

    return mean_reward, std_reward


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Experiment with an RL model.")
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

    args = parser.parse_args()

    params = json.loads(args.params)

    mean_reward, std_reward = experiment_model(
        params, args.env_name, args.total_timesteps
    )

    print(f"Experiment completed. Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


# python experiment.py --params '{"learning_rate": 0.001}' --env_name LunarLander-v2 --total_timesteps 100000

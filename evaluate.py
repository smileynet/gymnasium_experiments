import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def evaluate_model(model_path, env_name, n_eval_episodes=100, render=False):
    # Load the trained model
    model = PPO.load(model_path)

    # Create the environment
    env = gym.make(env_name, render_mode="human" if render else None)

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, render=render)

    print(f"Evaluation results for model: {model_path}")
    print(f"Environment: {env_name}")
    print(f"Number of evaluation episodes: {n_eval_episodes}")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    return mean_reward, std_reward

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the saved model")
    parser.add_argument("--env", type=str, required=True, help="Name of the Gym environment")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render the environment during evaluation")

    args = parser.parse_args()

    evaluate_model(args.model, args.env, args.episodes, args.render)

if __name__ == "__main__":
    main()

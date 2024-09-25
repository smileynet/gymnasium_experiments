import os

import gymnasium as gym
import numpy as np
import torch
from dotenv import load_dotenv
from stable_baselines3 import PPO
from torch import nn


class Agent(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observations):
        # Convert observations to the correct format
        if isinstance(observations, np.ndarray):
            observations = torch.FloatTensor(observations)
        elif isinstance(observations, tuple):
            observations = torch.FloatTensor(observations[0])

        # Ensure observations are 2D
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)

        # Get action from policy
        actions, _, _ = self.policy.forward(observations)
        return actions


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path, device="cpu")
    print(f"Model loaded from {model_path}")
    return model


def generate_agent_pt(model_path, env_id, output_path="agent.pt"):
    model = load_model(model_path)
    env = gym.make(env_id)

    agent = Agent(model.policy).cpu()

    # Get a sample observation from the environment
    sample_obs = env.observation_space.sample()

    # Convert the sample observation to a tensor
    if isinstance(sample_obs, np.ndarray):
        dummy_input = torch.FloatTensor(sample_obs).unsqueeze(0)
    elif isinstance(sample_obs, dict):
        dummy_input = {
            k: torch.FloatTensor(v).unsqueeze(0) for k, v in sample_obs.items()
        }
    else:
        raise ValueError(f"Unsupported observation type: {type(sample_obs)}")

    print(
        f"Dummy input shape: {dummy_input.shape if isinstance(dummy_input, torch.Tensor) else {k: v.shape for k, v in dummy_input.items()}}"
    )

    # Trace and freeze the agent
    try:
        traced_agent = torch.jit.trace(agent.eval(), dummy_input)
        frozen_agent = torch.jit.freeze(traced_agent)

        torch.jit.save(frozen_agent, output_path)
        print(f"Agent saved as TorchScript module: {output_path}")

        # Run test immediately after saving
        success = test_loaded_agent(output_path, env_id)
        return success
    except Exception as e:
        print(f"Error during tracing or saving: {str(e)}")
        return False


def test_loaded_agent(output_path, env_id):
    try:
        # Load the saved agent
        loaded_agent = torch.jit.load(output_path)
        print("Agent loaded successfully")

        # Create environment and get a sample observation
        env = gym.make(env_id)
        sample_obs = env.observation_space.sample()

        # Convert the sample observation to a tensor
        if isinstance(sample_obs, np.ndarray):
            dummy_input = torch.FloatTensor(sample_obs).unsqueeze(0)
        elif isinstance(sample_obs, dict):
            dummy_input = {
                k: torch.FloatTensor(v).unsqueeze(0) for k, v in sample_obs.items()
            }
        else:
            raise ValueError(f"Unsupported observation type: {type(sample_obs)}")

        # Test the loaded agent with dummy input
        with torch.no_grad():
            output = loaded_agent(dummy_input)

        print("Agent successfully processed dummy input")
        print(f"Output shape: {output.shape}")
        print(f"Output: {output}")

        return True
    except Exception as e:
        print(f"Error testing loaded agent: {str(e)}")
        return False


if __name__ == "__main__":
    load_dotenv()

    model_dir = os.getenv("MODEL_DIR", "models")
    best_model_name = os.getenv("BEST_MODEL_NAME", "best_model.zip")
    env_id = os.getenv("ENV_NAME", "LunarLander-v2")

    model_path = os.path.join(model_dir, best_model_name)
    output_path = "agent.pt"

    success = generate_agent_pt(model_path, env_id, output_path)

    if success:
        print("Agent generation and loading test passed successfully!")
    else:
        print("Agent generation and loading test failed.")

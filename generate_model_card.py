import os
import logging
from dotenv import load_dotenv
from utils import parse_arguments

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def generate_model_card(model_name, env_id, mean_reward, std_reward, hyperparameters):
    """
    Generate a model card in markdown format.

    Args:
        model_name (str): Name of the model (e.g., "PPO")
        env_id (str): ID of the environment (e.g., "LunarLander-v2")
        mean_reward (float): Mean reward from evaluation
        std_reward (float): Standard deviation of reward from evaluation
        hyperparameters (dict): Dictionary of hyperparameters used for training

    Returns:
        str: Markdown content of the model card

    Raises:
        ValueError: If HF_USERNAME is not set in the .env file
    """

    load_dotenv()

    username = os.getenv("HF_USERNAME")
    if not username:
        raise ValueError("HF_USERNAME not set in .env file")

    hyperparams_str = "{\n"
    for key, value in hyperparameters.items():
        if isinstance(value, str):
            hyperparams_str += f"    '{key}': '{value}',\n"
        else:
            hyperparams_str += f"    '{key}': {value},\n"
    hyperparams_str += "}"

    model_card = f"""
---
library_name: stable-baselines3
tags:
- {env_id}
- deep-reinforcement-learning
- reinforcement-learning
- gymnasium
model-index:
- name: {model_name}
  results:
  - task:
      type: reinforcement-learning
      name: {env_id}
    metrics:
      - type: mean_reward
        value: {mean_reward:.2f} +/- {std_reward:.2f}
        name: mean_reward
---

# {model_name} Agent playing {env_id}

This is a trained model of a {model_name} agent playing {env_id} using the [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) library.

## Usage

To use this model with Stable-Baselines3, follow these steps:

```python
import gymnasium as gym
from stable_baselines3 import {model_name}

# Create the environment
env = gym.make("{env_id}")

# Load the trained model
model = {model_name}.load("path/to/model.zip")

# Run the model
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Environment
The {env_id} environment is part of the [Gymnasium](https://gymnasium.farama.org/) library. 

## Training
The model was trained using the following hyperparameters:

```python
{hyperparams_str}
```

### Results
The trained agent achieved a mean reward of {mean_reward:.2f} +/- {std_reward:.2f} over {os.getenv('N_EVAL_EPISODES', '10')} evaluation episodes.
"""

    return model_card


if __name__ == "main":
    try:
        load_dotenv()

        model_name = os.getenv("MODEL_NAME", "PPO")
        env_id = os.getenv("ENV_NAME", "LunarLander-v2")
        mean_reward = float(os.getenv("MEAN_REWARD", "200"))
        std_reward = float(os.getenv("STD_REWARD", "10"))

        params = parse_arguments()

        outputs_dir = os.getenv("OUTPUTS_DIR")
        if not outputs_dir:
            raise ValueError("OUTPUTS_DIR not set in .env file")

        env_output_dir = os.path.join(outputs_dir, env_id)

        os.makedirs(env_output_dir, exist_ok=True)

        model_card = generate_model_card(
            model_name, env_id, mean_reward, std_reward, params
        )

        readme_path = os.path.join(env_output_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(model_card)

        logging.info(f"Model card generated and saved as {readme_path}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

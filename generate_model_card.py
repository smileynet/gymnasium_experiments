import logging
import os

import yaml
from dotenv import load_dotenv

from submission_context import SubmissionContext
from utils import parse_arguments

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def generate_model_card(context):
    """
    Generate a model card in markdown format with metadata and update the context.

    Args:
        context (SubmissionContext): Context object containing all necessary information

    Raises:
        ValueError: If HF_USERNAME is not set in the context
    """

    if not context.hf_username:
        raise ValueError("HF_USERNAME not set in context")

    hyperparams_str = "{\n"
    for key, value in context.hyperparameters.items():
        if isinstance(value, str):
            hyperparams_str += f"    '{key}': '{value}',\n"
        else:
            hyperparams_str += f"    '{key}': {value},\n"
    hyperparams_str += "}"

    metadata_yaml = yaml.dump(context.metadata, default_flow_style=False)

    context.model_card = f"""
---
{metadata_yaml.strip()}
---

# {context.model_architecture} Agent playing {context.env_id}

This is a trained model of a {context.model_architecture} agent playing {context.env_id} using the [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) library.

## Usage

To use this model with Stable-Baselines3, follow these steps:

```python
import gymnasium as gym
from stable_baselines3 import {context.model_architecture}

# Create the environment
env = gym.make("{context.env_id}")

# Load the trained model
model = {context.model_architecture}.load("path/to/model.zip")

# Run the model
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
Environment
The {context.env_id} environment is part of the Gymnasium library.

Training
The model was trained using the following hyperparameters:

{hyperparams_str}
Results
The trained agent achieved a mean reward of {context.results['mean_reward']:.2f} +/- {context.results['std_reward']:.2f} over {context.n_eval_episodes} evaluation episodes. """


if __name__ == "__main__":
    try:
        load_dotenv()

        model_architecture = os.getenv("MODEL_NAME", "PPO")
        env_id = os.getenv("ENV_NAME", "LunarLander-v2")
        mean_reward = float(os.getenv("MEAN_REWARD", "200"))
        std_reward = float(os.getenv("STD_REWARD", "10"))
        n_eval_episodes = int(os.getenv("N_EVAL_EPISODES", "10"))

        params = parse_arguments()

        outputs_dir = os.getenv("OUTPUTS_DIR")
        if not outputs_dir:
            raise ValueError("OUTPUTS_DIR not set in .env file")

        # Create a real SubmissionContext
        context = SubmissionContext(
            temp_dir=outputs_dir,
            model=None,  # Not needed for model card generation
            results={"mean_reward": mean_reward, "std_reward": std_reward},
            env_id=env_id,
            outputs_dir=outputs_dir,
            video_name="model_evaluation",  # Default value
            model_dir=os.getenv("MODEL_DIR", "models"),
            best_model_name=os.getenv("BEST_MODEL_NAME", "best_model.zip"),
            model_architecture=model_architecture,
            n_eval_episodes=n_eval_episodes,
            hf_token=os.getenv("HF_TOKEN"),
            hf_username=os.getenv("HF_USERNAME"),
            hyperparameters=params,
        )

        context.metadata = {
            "library_name": "stable-baselines3",
            "tags": [
                context.env_id,
                "deep-reinforcement-learning",
                "reinforcement-learning",
                "gymnasium",
            ],
            "model-index": [
                {
                    "name": context.model_architecture,
                    "results": [
                        {
                            "task": {
                                "type": "reinforcement-learning",
                                "name": context.env_id,
                            },
                            "metrics": [
                                {
                                    "type": "mean_reward",
                                    "value": f"{context.results['mean_reward']:.2f} +/- {context.results['std_reward']:.2f}",
                                    "name": "mean_reward",
                                }
                            ],
                        }
                    ],
                }
            ],
        }

        # Generate the model card
        generate_model_card(context)

        # Write the model card to OUTPUT_DIR/README.md
        readme_path = os.path.join(outputs_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(context.model_card)

        logging.info(f"Model card generated and saved as {readme_path}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

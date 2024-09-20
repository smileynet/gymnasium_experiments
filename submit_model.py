import os
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from huggingface_sb3 import package_to_hub
from evaluate_model import evaluate_and_record

# Load environment variables
load_dotenv()

# Get the username from .env file
username = os.getenv("HF_USERNAME")
if not username:
    raise ValueError("HF_USERNAME not set in .env file")

# Retrieve the environment ID from the environment variable
env_id = os.getenv("ENV_NAME", "LunarLander-v2")  # Default to "LunarLander-v2" if not set
output_dir = os.getenv("OUTPUTS", "outputs")

# Create the environment
env = make_vec_env(env_id, n_envs=1)

# Load the best model
model_path = os.path.join("models", "best_model.zip")
model = PPO.load(model_path)

# Define model architecture
model_architecture = "PPO"

# Create repo ID
repo_id = f"{username}/{model_architecture}-{env_id}"

# Evaluate and record
mean_reward, std_reward = evaluate_and_record(model, env_id, output_dir=output_dir)

# Ask user if they want to submit to Hugging Face Hub
submit = input("Do you want to submit the model to Hugging Face Hub? (y/n): ").lower()

if submit == "y":
    # This method save, evaluate, generate a model card and record a replay video of your agent before pushing the repo to the hub
    package_to_hub(
        model=model,
        model_name=f"{model_architecture}-{env_id}",
        model_architecture=model_architecture,
        env_id=env_id,
        eval_env=env,
        repo_id=repo_id,
        commit_message="Submitting best model",
    )
    print("Model submitted to Hugging Face Hub")
else:
    print("Model submission cancelled")

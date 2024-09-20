import os
import shutil

from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from evaluate_model import evaluate_and_record
from generate_model_card import generate_model_card

# Load environment variables
load_dotenv()

# Get the username from .env file
username = os.getenv("HF_USERNAME")
if not username:
    raise ValueError("HF_USERNAME not set in .env file")

# Retrieve the environment ID from the environment variable
env_id = os.getenv(
    "ENV_NAME", "LunarLander-v2"
)  # Default to "LunarLander-v2" if not set
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
    # Create a temporary directory for the model files
    temp_dir = "temp_model_dir"
    os.makedirs(temp_dir, exist_ok=True)

    # Save the model
    model.save(os.path.join(temp_dir, "model.zip"))

    # Copy the video
    video_path = os.path.join(output_dir, "videos", f"{env_id}-preview-episode-0.mp4")
    shutil.copy(video_path, os.path.join(temp_dir, "replay.mp4"))

    # Generate and save the model card
    model_card = generate_model_card(
        model_name=model_architecture,
        env_id=env_id,
        mean_reward=mean_reward,
        std_reward=std_reward,
        training_steps=model.num_timesteps,  # Assuming your model tracks this
        learning_rate=model.learning_rate,  # Assuming your model exposes this
    )
    with open(os.path.join(temp_dir, "README.md"), "w") as f:
        f.write(model_card)

    # Push to hub
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=temp_dir,
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"Model submitted to Hugging Face Hub: https://huggingface.co/{repo_id}")

    # Clean up temporary directory
    shutil.rmtree(temp_dir)
else:
    print("Model submission cancelled")

import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv


class SubmissionContext:
    def __init__(self):
        load_dotenv()
        self.temp_dir = os.getenv("TEMP_DIR", "temp")
        self.env_id = os.getenv("ENV_NAME", "LunarLander-v2")
        self.outputs_dir = os.getenv("OUTPUTS_DIR", "outputs")
        self.video_name = os.getenv("STUDY_NAME", "model_evaluation")
        self.model_dir = os.getenv("MODEL_DIR", "models")
        self.best_model_name = os.getenv("BEST_MODEL_NAME", "best_model.zip")
        self.n_eval_episodes = int(os.getenv("N_EVAL_EPISODES", 10))
        self.hf_token = os.getenv("HF_TOKEN")
        self.hf_username = os.getenv("HF_USERNAME")
        self.model_architecture = os.getenv("MODEL_ARCHITECTURE", "PPO")
        self.hyperparameters = {}
        self.metadata = {}

    def get_model_path(self) -> str:
        """Get the full path to the best model file."""
        return os.path.join(self.model_dir, self.best_model_name)

    def get_results_path(self) -> str:
        """Get the full path to the results JSON file."""
        return os.path.join(self.outputs_dir, "results.json")

    def get_video_dir(self) -> str:
        """Get the directory containing video files."""
        return os.path.join(self.outputs_dir, "videos")

    def get_model_name(self) -> str:
        """Get the name of the model."""
        return f"{self.model_architecture.lower()}-{self.env_id}"

    def get_repo_id(self) -> str:
        """Get the repository ID for Hugging Face Hub."""
        model_name = self.get_model_name()
        return f"{self.hf_username}/{model_name}"

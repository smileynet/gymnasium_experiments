import os
from typing import Dict, Any

from dotenv import load_dotenv
from rich.table import Table

from logging_config import console, logger


class SubmissionContext:
    """
    A class to manage and display the context for model submission.

    This class loads environment variables, sets up configuration parameters,
    and provides methods to access various paths and identifiers related to
    the model submission process.
    """

    def __init__(self):
        """
        Initialize the SubmissionContext with values from environment variables.
        """
        try:
            load_dotenv()
            self.temp_dir: str = os.getenv("TEMP_DIR", "temp")
            self.env_id: str = os.getenv("ENV_NAME", "LunarLander-v2")
            self.outputs_dir: str = os.getenv("OUTPUTS_DIR", "outputs")
            self.video_name: str = os.getenv("STUDY_NAME", "model_evaluation")
            self.model_dir: str = os.getenv("MODEL_DIR", "models")
            self.best_model_name: str = os.getenv("BEST_MODEL_NAME", "best_model.zip")
            self.n_eval_episodes: int = int(os.getenv("N_EVAL_EPISODES", 10))
            self.hf_token: str = os.getenv("HF_TOKEN")
            self.hf_username: str = os.getenv("HF_USERNAME")
            self.model_architecture: str = os.getenv("MODEL_ARCHITECTURE", "PPO")
            self.hyperparameters: Dict[str, Any] = {}
            self.metadata: Dict[str, Any] = {}

            self.display_context()
        except Exception as e:
            logger.error(f"Error initializing SubmissionContext: {str(e)}")
            raise

    def display_context(self) -> None:
        """
        Display the current context settings in a formatted table.
        """
        try:
            table = Table(title="Submission Context")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="magenta")

            for key, value in self.__dict__.items():
                if not key.startswith("_"):
                    table.add_row(key, str(value))

            console.print(table)
            logger.debug("Submission context displayed successfully")
        except Exception as e:
            logger.error(f"Error displaying submission context: {str(e)}")

    def get_model_path(self) -> str:
        """
        Get the full path to the best model file.

        Returns:
            str: The full path to the best model file.
        """
        try:
            path = os.path.join(self.model_dir, self.best_model_name)
            logger.debug(f"Model path: {path}")
            return path
        except Exception as e:
            logger.error(f"Error getting model path: {str(e)}")
            raise

    def get_results_path(self) -> str:
        """
        Get the full path to the results JSON file.

        Returns:
            str: The full path to the results JSON file.
        """
        try:
            path = os.path.join(self.outputs_dir, "results.json")
            logger.debug(f"Results path: {path}")
            return path
        except Exception as e:
            logger.error(f"Error getting results path: {str(e)}")
            raise

    def get_video_dir(self) -> str:
        """
        Get the directory containing video files.

        Returns:
            str: The path to the directory containing video files.
        """
        try:
            path = os.path.join(self.outputs_dir, "videos")
            logger.debug(f"Video directory: {path}")
            return path
        except Exception as e:
            logger.error(f"Error getting video directory: {str(e)}")
            raise

    def get_model_name(self) -> str:
        """
        Get the name of the model.

        Returns:
            str: The name of the model.
        """
        try:
            name = f"{self.model_architecture.lower()}-{self.env_id}"
            logger.debug(f"Model name: {name}")
            return name
        except Exception as e:
            logger.error(f"Error getting model name: {str(e)}")
            raise

    def get_repo_id(self) -> str:
        """
        Get the repository ID for Hugging Face Hub.

        Returns:
            str: The repository ID for Hugging Face Hub.
        """
        try:
            model_name = self.get_model_name()
            repo_id = f"{self.hf_username}/{model_name}"
            logger.debug(f"Repository ID: {repo_id}")
            return repo_id
        except Exception as e:
            logger.error(f"Error getting repository ID: {str(e)}")
            raise

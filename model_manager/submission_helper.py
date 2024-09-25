import json
import os
import shutil
import zipfile

import stable_baselines3
from huggingface_hub import HfApi, login
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from evaluate_model import evaluate_and_record
from generate_model_card import generate_model_card
from logging_config import console, logger
from submission_context import SubmissionContext
from utils import get_latest_video, parse_arguments


class SubmissionHelper:
    """
    A helper class to manage the submission process for trained models.
    """

    def __init__(self):
        """
        Initialize the SubmissionHelper with a SubmissionContext.
        """
        self.context = SubmissionContext()

    def load_model(self, model_path=None):
        """
        Load the trained model from the specified path.

        Args:
            model_path (str, optional): Path to the model file. If None, uses the default path from context.

        Returns:
            stable_baselines3.PPO: The loaded model.

        Raises:
            FileNotFoundError: If the model file is not found at the specified path.
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[green]Loading model...", total=None)
            if model_path is None:
                model_path = self.context.get_model_path()

            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")

            try:
                logger.debug(f"Loading model from {model_path}")
                model = PPO.load(model_path)
                logger.debug(f"Model loaded from {model_path}")
                self.context.model = model
                self.context.hyperparameters = parse_arguments()
                progress.update(task, completed=True)
                logger.info("Model loaded successfully")
                return model
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise

    def prepare_submission_files(self) -> bool:
        """
        Prepare all necessary files for submission in the temporary directory.

        Returns:
            bool: True if preparation was successful, False otherwise.
        """
        try:
            self._copy_model_zip()
            self._copy_results_json()
            self._convert_and_copy_video()
            self._create_metadata()
            self._generate_model_card()
            self._generate_config()
            return True
        except Exception as e:
            logger.error(f"Error preparing submission files: {str(e)}")
            return False

    def _create_metadata(self):
        """
        Create metadata for the model submission and display it.
        """
        from rich.table import Table

        try:
            self.context.metadata = {
                "library_name": "stable-baselines3",
                "tags": [
                    self.context.env_id,
                    "deep-reinforcement-learning",
                    "reinforcement-learning",
                    "gymnasium",
                ],
                "model-index": [
                    {
                        "name": self.context.model_architecture,
                        "results": [
                            {
                                "task": {
                                    "type": "reinforcement-learning",
                                    "name": self.context.env_id,
                                },
                                "metrics": [
                                    {
                                        "type": "mean_reward",
                                        "value": f"{self.context.results['mean_reward']:.2f} +/- {self.context.results['std_reward']:.2f}",
                                        "name": "mean_reward",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }

            table = Table(title="Metadata")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="magenta")

            for key, value in self.context.metadata.items():
                table.add_row(str(key), str(value))

            console.print(table)
            logger.debug("Metadata created and displayed")
        except Exception as e:
            logger.error(f"Error creating metadata: {str(e)}")
            raise

    def _generate_model_card(self):
        """
        Generate and save the model card.
        """
        try:
            generate_model_card(self.context)
            with open(os.path.join(self.context.temp_dir, "README.md"), "w") as f:
                f.write(self.context.model_card)
            logger.debug(
                f"Generated and saved model card to {self.context.temp_dir}/README.md"
            )
        except Exception as e:
            logger.error(f"Error generating model card: {str(e)}")
            raise

    def validate_submission_files(self) -> bool:
        """
        Validate that all required files are present in the temporary directory.
        Returns:
            bool: True if all required files are present, False otherwise.
        """
        model_name = self.context.get_model_name() + ".zip"
        required_files = [
            model_name,
            "results.json",
            "replay.mp4",
            "README.md",
            "config.json",
        ]
        missing_files = [
            file
            for file in required_files
            if not os.path.exists(os.path.join(self.context.temp_dir, file))
        ]

        if missing_files:
            logger.error(
                f"Missing required files for submission: {', '.join(missing_files)}"
            )
            return False

        logger.debug("All required files for submission are present")
        return True

    def submit_to_hub(self):
        """
        Submit the model and associated files to Hugging Face Hub.

        Raises:
            Exception: If an error occurs during submission.
        """
        try:
            login(token=self.context.hf_token)
            api = HfApi()
            repo_id = self.context.get_repo_id()

            tags = self.context.metadata.get("tags", [])

            logger.debug(
                f"Creating repository {repo_id} on Hugging Face Hub with tags:\n {tags}"
            )

            api.create_repo(
                repo_id,
                exist_ok=True,
                repo_type="model",
            )
            api.upload_folder(
                folder_path=self.context.temp_dir,
                repo_id=repo_id,
                repo_type="model",
                path_in_repo=".",
                commit_message="Upload model files",
                commit_description="Uploading model files including README, and metadata",
            )
            logger.info(
                f"Model and results submitted to Hugging Face Hub: https://huggingface.co/{repo_id}"
            )
        except Exception as e:
            logger.error(
                f"An error occurred during submission to Hugging Face Hub: {str(e)}"
            )
            raise

    def _generate_config(self):
        """
        Generate a configuration file compatible with the submission format.
        """
        try:
            model_path = self.context.get_model_path()
            output_dir = self.context.temp_dir

            model_name = self.context.get_model_name()
            unzipped_model_dir = os.path.join(output_dir, model_name)
            os.makedirs(unzipped_model_dir, exist_ok=True)

            with zipfile.ZipFile(model_path, "r") as zip_ref:
                zip_ref.extractall(unzipped_model_dir)

            with open(os.path.join(unzipped_model_dir, "data")) as json_file:
                data = json.load(json_file)
                data["system_info"] = stable_baselines3.get_system_info(
                    print_info=False
                )[0]

            with open(os.path.join(output_dir, "config.json"), "w") as outfile:
                json.dump(data, outfile)

            logger.debug(f"config.json generated at {self.context.temp_dir}")
        except Exception as e:
            logger.error(f"Error generating config: {str(e)}")
            raise

    def _copy_model_zip(self):
        """
        Copy the model zip file to the temporary directory.
        """
        try:
            model_path = self.context.get_model_path()
            os.makedirs(self.context.temp_dir, exist_ok=True)
            model_name = self.context.get_model_name() + ".zip"
            shutil.copy(
                model_path,
                os.path.join(self.context.temp_dir, model_name),
            )
            logger.debug(f"Copied model zip to {self.context.temp_dir}")
        except Exception as e:
            logger.error(f"Error copying model zip: {str(e)}")
            raise

    def _copy_results_json(self):
        """
        Copy the results JSON file to the temporary directory.
        """
        try:
            results_path = self.context.get_results_path()
            os.makedirs(self.context.temp_dir, exist_ok=True)
            if os.path.exists(results_path):
                shutil.copy(
                    results_path, os.path.join(self.context.temp_dir, "results.json")
                )
                logger.debug(f"Copied results.json to {self.context.temp_dir}")
            else:
                logger.warning(f"results.json not found at {results_path}")
        except Exception as e:
            logger.error(f"Error copying results json: {str(e)}")
            raise

    def _convert_and_copy_video(self):
        """
        Convert and copy the latest video file to the temporary directory.
        """
        try:
            video_dir = self.context.get_video_dir()
            latest_video = get_latest_video(video_dir, self.context.video_name)
            os.makedirs(self.context.temp_dir, exist_ok=True)
            if latest_video:
                output_path = os.path.join(self.context.temp_dir, "replay.mp4")
                shutil.copy(latest_video, output_path)
                logger.debug(f"Converted and copied video to {output_path}")
            else:
                logger.warning(
                    f"No matching video file found for '{self.context.video_name}' in {video_dir}"
                )
        except Exception as e:
            logger.error(f"Error converting and copying video: {str(e)}")
            raise

    def evaluate_and_record(self):
        """
        Evaluate the model and record the results.

        Returns:
            dict: The evaluation results.

        Raises:
            Exception: If an error occurs during evaluation and recording.
        """
        env = make_vec_env(self.context.env_id, n_envs=1)
        try:
            results = evaluate_and_record(
                model=self.context.model,
                env_id=self.context.env_id,
                video_name=self.context.video_name,
                n_eval_episodes=self.context.n_eval_episodes,
                video_length=1000,
            )
            self.context.results = results
            console.print(
                "Be sure to double check the video and results to ensure the right model was loaded.",
                style="bold bright_magenta",
            )
            return results
        except Exception as e:
            logger.error(f"Error during evaluation and recording: {str(e)}")
            raise
        finally:
            env.close()

    def prompt_for_submission(self) -> bool:
        """
        Prompt the user to confirm submission to Hugging Face Hub.

        Returns:
            bool: True if the user confirms submission, False otherwise.
        """
        submit = Confirm.ask(
            "[bright_yellow]Do you want to submit the model to Hugging Face Hub?[/bright_yellow]",
            default=True,
        )
        return submit

    def cleanup(self):
        """
        Clean up the temporary directory after submission.
        """
        try:
            shutil.rmtree(self.context.temp_dir)
            logger.debug(f"Cleaned up temporary directory: {self.context.temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory: {str(e)}")

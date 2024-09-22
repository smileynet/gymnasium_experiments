import logging
import os
import shutil

from huggingface_hub import HfApi, login
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from evaluate_model import evaluate_and_record
from generate_agent import generate_agent_pt
from generate_model_card import generate_model_card
from submission_context import SubmissionContext
from utils import convert_video, get_latest_video, parse_arguments


class SubmissionHelper:
    def __init__(self):
        self.context = SubmissionContext()

    def load_model(self, model_path=None):
        """Load the trained model from the specified path."""

        if model_path is None:
            model_path = self.context.get_model_path()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logging.debug(f"Loading model from {model_path}")
        model = PPO.load(model_path)
        logging.debug(f"Model loaded from {model_path}")

        self.context.model = model
        self.context.hyperparameters = parse_arguments()

        return model

    def prepare_submission_files(self) -> bool:
        """Prepare all necessary files for submission in the temporary directory."""
        try:
            self._copy_model_zip()
            self._copy_results_json()
            self._convert_and_copy_video()
            self._create_metadata()
            self._generate_model_card()
            self._generate_agent_pt()
            return True
        except Exception as e:
            logging.error(f"Error preparing submission files: {str(e)}")
            return False

    def _create_metadata(self):
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

    def _generate_model_card(self):
        generate_model_card(self.context)
        with open(os.path.join(self.context.temp_dir, "README.md"), "w") as f:
            f.write(self.context.model_card)
        logging.info(
            f"Generated and saved model card to {self.context.temp_dir}/README.md"
        )

    def _generate_agent_pt(self):
        model_path = self.context.get_model_path()
        os.makedirs(self.context.temp_dir, exist_ok=True)
        output_path = os.path.join(self.context.temp_dir, "agent.pt")
        success = generate_agent_pt(model_path, self.context.env_id, output_path)
        if not success:
            raise RuntimeError("Failed to generate agent.pt")
        logging.info(f"Generated agent.pt at {output_path}")

    def validate_submission_files(self) -> bool:
        """Validate that all required files are present in the temporary directory."""
        required_files = [
            self.context.best_model_name,
            "agent.pt",
            "results.json",
            "replay.mp4",
            "README.md",
        ]
        missing_files = [
            file
            for file in required_files
            if not os.path.exists(os.path.join(self.context.temp_dir, file))
        ]

        if missing_files:
            logging.error(
                f"Missing required files for submission: {', '.join(missing_files)}"
            )
            return False

        logging.info("All required files for submission are present")
        return True

    def submit_to_hub(self):
        """Submit the model and associated files to Hugging Face Hub."""
        try:
            self.validate_submission_files()
        except Exception as e:
            logging.error(f"Error validating submission files: {str(e)}")
            raise

        try:
            login(token=self.context.hf_token)
            api = HfApi()
            repo_id = self.context.get_repo_id()

            tags = self.context.metadata.get("tags", [])

            logging.info(
                f"Creating repository {repo_id} on Hugging Face Hub with tags:\n {tags}"
            )

            # Use the metadata from the context
            api.create_repo(
                repo_id,
                exist_ok=True,
                repo_type="model",
            )
            api.upload_folder(
                folder_path=self.context.temp_dir,
                repo_id=repo_id,
                repo_type="model",
                commit_message="Upload model files",
                commit_description="Uploading model files including README, agent.pt, and metadata",
                ignore_patterns=[".*"],
            )

            logging.info(
                f"Model and results submitted to Hugging Face Hub: https://huggingface.co/{repo_id}"
            )
        except Exception as e:
            logging.error(
                f"An error occurred during submission to Hugging Face Hub: {str(e)}"
            )
            raise

    def _copy_model_zip(self):
        model_path = self.context.get_model_path()
        os.makedirs(self.context.temp_dir, exist_ok=True)
        if os.path.exists(model_path):
            shutil.copy(
                model_path,
                os.path.join(self.context.temp_dir, self.context.best_model_name),
            )
            logging.info(f"Copied model zip to {self.context.temp_dir}")
        else:
            raise FileNotFoundError(f"Model zip not found at {model_path}")

    def _copy_results_json(self):
        results_path = self.context.get_results_path()
        os.makedirs(self.context.temp_dir, exist_ok=True)
        if os.path.exists(results_path):
            shutil.copy(
                results_path, os.path.join(self.context.temp_dir, "results.json")
            )
            logging.info(f"Copied results.json to {self.context.temp_dir}")
        else:
            logging.warning(f"results.json not found at {results_path}")

    def _convert_and_copy_video(self):
        video_dir = self.context.get_video_dir()
        latest_video = get_latest_video(video_dir, self.context.video_name)
        os.makedirs(self.context.temp_dir, exist_ok=True)
        if latest_video:
            output_path = os.path.join(self.context.temp_dir, "replay.mp4")
            convert_video(latest_video, output_path)
            logging.info(f"Converted and copied video to {output_path}")
        else:
            logging.warning(
                f"No matching video file found for '{self.context.video_name}' in {video_dir}"
            )

    def evaluate_and_record(self):
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
            print(
                "Be sure to double check the video and results to ensure the right model was loaded."
            )
            return results
        except Exception as e:
            logging.error(f"Error during evaluation and recording: {str(e)}")
        finally:
            env.close()

    def prompt_for_submission(self) -> bool:
        """Prompt the user to confirm submission to Hugging Face Hub."""
        submit = input(
            "Do you want to submit the model to Hugging Face Hub? (y/n): "
        ).lower()
        return submit == "y"

    def cleanup(self):
        """Clean up the temporary directory after submission."""
        shutil.rmtree(self.context.temp_dir)

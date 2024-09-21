import logging
import os
import shutil

from huggingface_hub import HfApi, login
from stable_baselines3 import PPO

from generate_agent import generate_agent_pt
from generate_model_card import generate_model_card
from submission_context import SubmissionContext
from utils import convert_video, get_latest_video


class SubmissionHelper:
    @staticmethod
    def load_model(context: SubmissionContext):
        """Load the trained model from the specified path."""
        model_path = context.get_model_path()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logging.info(f"Loading model from {model_path}")
        model = PPO.load(model_path)
        logging.info(f"Model loaded from {model_path}")
        context.model = model
        context.hyperparameters = model.get_parameters()
        return model

    @staticmethod
    def prepare_submission_files(context: SubmissionContext) -> bool:
        """Prepare all necessary files for submission in the temporary directory."""
        try:
            SubmissionHelper._copy_model_zip(context)
            SubmissionHelper._copy_results_json(context)
            SubmissionHelper._convert_and_copy_video(context)
            SubmissionHelper.create_metadata(context)  # Add this line
            SubmissionHelper._generate_model_card(context)
            SubmissionHelper._generate_agent_pt(context)
            return True
        except Exception as e:
            logging.error(f"Error preparing submission files: {str(e)}")
            return False

    @staticmethod
    def create_metadata(context: SubmissionContext):
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

    @staticmethod
    def _generate_model_card(context: SubmissionContext):
        generate_model_card(context)
        with open(os.path.join(context.temp_dir, "README.md"), "w") as f:
            f.write(context.model_card)
        logging.info(f"Generated and saved model card to {context.temp_dir}/README.md")

    @staticmethod
    def _generate_agent_pt(context: SubmissionContext):
        model_path = context.get_model_path()
        output_path = os.path.join(context.temp_dir, "agent.pt")
        success = generate_agent_pt(model_path, context.env_id, output_path)
        if not success:
            raise RuntimeError("Failed to generate agent.pt")
        logging.info(f"Generated agent.pt at {output_path}")

    @staticmethod
    def validate_submission_files(context: SubmissionContext) -> bool:
        """Validate that all required files are present in the temporary directory."""
        required_files = [
            context.best_model_name,
            "agent.pt",
            "results.json",
            "replay.mp4",
            "README.md",
        ]
        missing_files = [
            file
            for file in required_files
            if not os.path.exists(os.path.join(context.temp_dir, file))
        ]

        if missing_files:
            logging.error(
                f"Missing required files for submission: {', '.join(missing_files)}"
            )
            return False

        logging.info("All required files for submission are present")
        return True

    @staticmethod
    def submit_to_hub(context: SubmissionContext):
        """Submit the model and associated files to Hugging Face Hub."""
        try:
            login(token=context.hf_token)
            api = HfApi()
            repo_id = context.get_repo_id()

            # Use the metadata from the context
            api.create_repo(
                repo_id,
                exist_ok=True,
                repo_type="model",
                metadata={context.metadata["tags"]},
            )
            api.upload_folder(
                folder_path=context.temp_dir,
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

    @staticmethod
    def _copy_model_zip(context: SubmissionContext):
        model_path = context.get_model_path()
        if os.path.exists(model_path):
            shutil.copy(
                model_path, os.path.join(context.temp_dir, context.best_model_name)
            )
            logging.info(f"Copied model zip to {context.temp_dir}")
        else:
            raise FileNotFoundError(f"Model zip not found at {model_path}")

    @staticmethod
    def _copy_results_json(context: SubmissionContext):
        results_path = context.get_results_path()
        if os.path.exists(results_path):
            shutil.copy(results_path, os.path.join(context.temp_dir, "results.json"))
            logging.info(f"Copied results.json to {context.temp_dir}")
        else:
            logging.warning(f"results.json not found at {results_path}")

    @staticmethod
    def _convert_and_copy_video(context: SubmissionContext):
        video_dir = context.get_video_dir()
        latest_video = get_latest_video(video_dir, context.video_name)
        if latest_video:
            output_path = os.path.join(context.temp_dir, "replay.mp4")
            convert_video(latest_video, output_path)
            logging.info(f"Converted and copied video to {output_path}")
        else:
            logging.warning(
                f"No matching video file found for '{context.video_name}' in {video_dir}"
            )

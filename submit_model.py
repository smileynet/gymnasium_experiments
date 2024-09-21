import logging
import os
import shutil

from stable_baselines3.common.env_util import make_vec_env

from evaluate_model import evaluate_and_record
from generate_agent import generate_agent_pt
from submission_helper import SubmissionContext, SubmissionHelper


def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def main():
    """Main function to handle model submission process."""
    setup_logging()

    try:
        temp_dir = "temp_model_dir"
        os.makedirs(temp_dir, exist_ok=True)

        context = SubmissionContext.from_env(temp_dir)

        if not context.hf_token or not context.hf_username:
            raise ValueError("HF_TOKEN or HF_USERNAME not set in .env file")

        env = make_vec_env(context.env_id, n_envs=1)
        model = SubmissionHelper.load_model(context)

        results = evaluate_and_record(
            model=context.model,
            env_id=context.env_id,
            video_name=context.video_name,
            n_eval_episodes=context.n_eval_episodes,
            video_length=1000,  # You might want to add this to SubmissionContext if it varies
        )
        context.results = results
        print(
            "Be sure to double check the video and results to ensure the right model was loaded."
        )
        submit = input(
            "Do you want to submit the model to Hugging Face Hub? (y/n): "
        ).lower()

        if submit == "y":
            if not generate_agent_pt(
                context.get_model_path(),
                context.env_id,
                os.path.join(context.temp_dir, "agent.pt"),
            ):
                logging.error("Failed to generate agent.pt")
                return

            if not SubmissionHelper.prepare_submission_files(context):
                logging.error("Failed to prepare submission files")
                return

            if not SubmissionHelper.validate_submission_files(context):
                logging.error("Submission files validation failed")
                return

            try:
                SubmissionHelper.submit_to_hub(context)
            finally:
                shutil.rmtree(context.temp_dir)
        else:
            logging.info("Model submission cancelled")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()

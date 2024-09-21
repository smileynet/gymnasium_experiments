import logging
import os
import shutil

from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from evaluate_model import evaluate_and_record
from generate_model_card import generate_model_card
from utils import convert_video, get_latest_video, parse_arguments


def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def load_environment_variables():
    """Load environment variables and validate required ones."""
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not set in .env file")

    username = os.getenv("HF_USERNAME")
    if not username:
        raise ValueError("HF_USERNAME not set in .env file")

    return hf_token, username


def load_model(model_path):
    """Load the trained model from the specified path."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logging.info(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    logging.info(f"Model loaded from {model_path}")
    return model


def submit_to_hub(temp_dir, repo_id):
    """Submit the model and associated files to Hugging Face Hub."""
    try:
        api = HfApi()
        api.create_repo(repo_id, exist_ok=True)
        logging.info(f"Uploading files from {temp_dir} to {repo_id}")
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                logging.info(f"Uploading file: {file_path}")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file,
                    repo_id=repo_id,
                    repo_type="model",
                )
        logging.info(
            f"Model and results submitted to Hugging Face Hub: https://huggingface.co/{repo_id}"
        )
    except Exception as e:
        logging.error(
            f"An error occurred during submission to Hugging Face Hub: {str(e)}"
        )
        raise


def main():
    """Main function to handle model submission process."""
    setup_logging()

    try:
        hf_token, username = load_environment_variables()
        login(token=hf_token)

        env_id = os.getenv("ENV_NAME", "LunarLander-v2")
        model_dir = os.getenv("MODEL_DIR", "models")
        best_model_name = os.getenv("BEST_MODEL_NAME", "best_model.zip")
        outputs_dir = os.getenv("OUTPUTS_DIR", "outputs")
        n_eval_episodes = int(os.getenv("N_EVAL_EPISODES", 10))
        video_name = os.getenv("STUDY_NAME", "model_evaluation")

        env = make_vec_env(env_id, n_envs=1)
        model_path = os.path.join(model_dir, best_model_name)
        model = load_model(model_path)

        # TODO: Parameterize this
        model_architecture = "PPO"
        repo_id = f"{username}/{model_architecture}-{env_id}"

        results = evaluate_and_record(
            model, env_id, video_name, n_eval_episodes=n_eval_episodes
        )
        mean_reward = results["mean_reward"]
        std_reward = results["std_reward"]

        submit = input(
            "Do you want to submit the model to Hugging Face Hub? (y/n): "
        ).lower()

        if submit == "y":
            temp_dir = "temp_model_dir"
            os.makedirs(temp_dir, exist_ok=True)

            model.save(os.path.join(temp_dir, "model.zip"))

            results_path = os.path.join(outputs_dir, "results.json")
            if os.path.exists(results_path):
                shutil.copy(results_path, os.path.join(temp_dir, "results.json"))
            else:
                logging.warning(f"results.json not found at {results_path}")

            video_dir = os.path.join(outputs_dir, "videos")
            latest_video = get_latest_video(video_dir, video_name)

            if latest_video:
                logging.info(f"Found latest video: {latest_video}")
                output_path = os.path.join(temp_dir, "replay.mp4")
                try:
                    convert_video(latest_video, output_path)
                    logging.info(f"Video converted and saved to {output_path}")
                except Exception as e:
                    logging.error(f"Failed to convert video: {str(e)}")
            else:
                logging.warning(
                    f"No matching video file found for '{video_name}' in {video_dir}"
                )

            params = parse_arguments()

            model_card = generate_model_card(
                model_name=model_architecture,
                env_id=env_id,
                mean_reward=mean_reward,
                std_reward=std_reward,
                hyperparameters=params,
            )
            with open(os.path.join(temp_dir, "README.md"), "w") as f:
                f.write(model_card)

            try:
                submit_to_hub(temp_dir, repo_id)
            finally:
                shutil.rmtree(temp_dir)
        else:
            logging.info("Model submission cancelled")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()

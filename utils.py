import argparse
import glob
import json
import os

from dotenv import load_dotenv

from logging_config import logger


def get_latest_video(video_dir, video_name):
    """
    Get the most recently modified video file matching the given name.

    Args:
        video_dir (str): Directory to search for video files.
        video_name (str): Base name of the video files to search for.

    Returns:
        str: Path to the most recent video file, or None if no matching files found.
    """
    pattern = os.path.join(video_dir, f"{video_name}*.mp4")
    matching_files = glob.glob(pattern)

    if not matching_files:
        logger.warning(f"No video files found matching pattern: {pattern}")
        return None

    latest_file = max(matching_files, key=os.path.getmtime)
    logger.info(f"Latest video file found: {latest_file}")
    return latest_file

def parse_arguments():
    """
    Parse command line arguments and return model parameters.

    Returns:
        dict: A dictionary of model parameters.

    Raises:
        ValueError: If the provided JSON string for parameters is invalid.
    """
    parser = argparse.ArgumentParser(description="Train an RL model with parameters.")
    parser.add_argument("--params", type=str, help="JSON string of model parameters")
    args = parser.parse_args()

    if args.params:
        try:
            params = json.loads(args.params)
            logger.info("Using provided parameters")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON string for parameters: {e}")
            raise ValueError("Invalid JSON string for parameters") from e
    else:
        params = {
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
        logger.info("Using default parameters")

    logger.debug(f"Model parameters: {json.dumps(params, indent=2)}")
    return params

def get_db_url():
    load_dotenv()

    db_name = os.getenv('DB_NAME', 'optuna')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT', '3306')

    db_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    logger.debug(f"Database string: {db_string}")

    if all([db_user, db_password, db_host]):
        return db_string
    return None

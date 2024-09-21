import argparse
import glob
import json
import logging
import os
from datetime import datetime

import ffmpeg


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
        logging.warning(f"No video files found matching pattern: {pattern}")
        return None

    latest_file = max(matching_files, key=os.path.getmtime)
    logging.info(f"Latest video file found: {latest_file}")
    return latest_file


def convert_video(input_path, output_path):
    """
    Convert a video file to mp4 format using ffmpeg.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path where the converted video will be saved.

    Raises:
        ffmpeg.Error: If an error occurs during video conversion.
    """
    try:
        (
            ffmpeg.input(input_path)
            .output(output_path, vcodec="libx264", acodec="aac", strict="experimental")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        logging.info(f"Video successfully converted and saved to {output_path}")
    except ffmpeg.Error as e:
        logging.error(f"Error occurred during video conversion: {e.stderr.decode()}")
        raise


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
            logging.info("Using provided parameters")
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON string for parameters: {e}")
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
        logging.info("Using default parameters")

    logging.debug(f"Model parameters: {json.dumps(params, indent=2)}")
    return params


def setup_logging():
    """
    Set up logging configuration.

    Returns:
        logging.Logger: Configured logger object.
    """
    try:
        os.makedirs("./logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename=f"./logs/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
        return logging.getLogger(__name__)
    except Exception as e:
        print(f"Error setting up logging: {e}")
        raise

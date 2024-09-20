# train.py

import os
from dotenv import load_dotenv
from model import create_model, load_model, train_model
from utils import parse_arguments, setup_logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()
logger = setup_logging()

# Get the log and model directories from environment variables
LOG_DIR = os.getenv("LOG_DIR", "./logs")
MODEL_DIR = os.getenv("MODEL_DIR", "./models")


def main():
    try:
        params = parse_arguments()

        saved_model_path = os.path.join(
            MODEL_DIR, os.getenv("BEST_MODEL_NAME", "best_model.zip")
        )

        if os.path.exists(saved_model_path):
            model, env, eval_env = load_model(saved_model_path)
            logger.info(
                f"Continuing training with loaded model from {saved_model_path}."
            )
        else:
            model, env, eval_env = create_model(params)
            logger.info("Starting training from scratch.")

        model, best_mean_reward = train_model(model, env, eval_env, params)
        logger.info(f"Training completed. Best mean reward: {best_mean_reward:.2f}")

        # Save the final model
        final_model_path = os.path.join(MODEL_DIR, "final_model.zip")
        model.save(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()

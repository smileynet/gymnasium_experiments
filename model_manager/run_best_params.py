import argparse
import logging
import sys

import optuna
from dotenv import load_dotenv
from train import train_model

from utils import get_db_url

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_best_params(study_name, storage=None):
    try:
        if storage is None:
            storage = get_db_url()
            if storage:
                logger.info("Using database URL from environment variables.")
            else:
                raise ValueError("No storage provided and no valid environment variables found.")
        else:
            logger.info("Using provided storage URL.")

        study = optuna.load_study(study_name=study_name, storage=storage)
        best_trial = study.best_trial
        best_params = best_trial.params
        logger.info(f"Successfully loaded best parameters for study: {study_name}")
        return best_params
    except ValueError as e:
        logger.error(f"Error with study storage: {str(e)}")
        raise
    except KeyError:
        logger.error(f"Study not found: {study_name}")
        raise
    except Exception as e:
        logger.error(f"Error loading best parameters: {str(e)}")
        raise

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run training with best parameters from an Optuna study."
    )
    parser.add_argument(
        "--study_name", type=str, required=True, help="Name of the Optuna study"
    )
    parser.add_argument(
        "--storage", type=str, help="Storage URL for the Optuna study (overrides environment variables if provided)"
    )
    return parser.parse_args()

def run_training_with_best_params(args):
    try:
        best_params = load_best_params(args.study_name, args.storage)
        logger.info(f"Best parameters: {best_params}")

        logger.info("Starting training process...")
        model, best_mean_reward = train_model(best_params)
        logger.info(
            f"Training completed successfully. Best mean reward: {best_mean_reward:.2f}"
        )
        return model, best_mean_reward
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

def main():
    args = parse_arguments()
    try:
        model, best_mean_reward = run_training_with_best_params(args)
        print(f"Best mean reward {best_mean_reward} for model {model}")
        return 0
    except Exception:
        return 1

if __name__ == "__main__":
    sys.exit(main())

import optuna
import argparse
import logging
import sys
from train import train_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_best_params(study_name, storage=None):
    try:
        if storage:
            study = optuna.load_study(study_name=study_name, storage=storage)
        else:
            raise ValueError("Storage must be provided to load an existing study.")

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
        "--storage", type=str, required=True, help="Storage URL for the Optuna study"
    )
    parser.add_argument(
        "--env_name", type=str, required=True, help="Name of the Gym environment"
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        required=True,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save the trained model"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Number of evaluations with no improvement after which training will be stopped",
    )
    return parser.parse_args()


def run_training_with_best_params(args):
    try:
        best_params = load_best_params(args.study_name, args.storage)
        logger.info(f"Best parameters: {best_params}")

        logger.info("Starting training process...")
        model, best_mean_reward = train_model(
            best_params,
            args.env_name,
            args.total_timesteps,
            args.save_path,
            n_epochs=args.n_epochs,
            early_stopping_patience=args.early_stopping_patience,
        )
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
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())

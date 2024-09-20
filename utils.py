import argparse
import json
import logging
import os
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train an RL model with parameters.")
    parser.add_argument("--params", type=str, help="JSON string of model parameters")
    args = parser.parse_args()

    if args.params:
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string for parameters")
    else:
        # Default parameters
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
        print("Using default parameters:", json.dumps(params, indent=2))

    return params


def setup_logging():
    os.makedirs("./logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)

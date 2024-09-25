import os

import matplotlib.pyplot as plt
import optuna
from dotenv import load_dotenv
from optuna.visualization import (
    plot_contour,
    plot_edf,
    plot_intermediate_values,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_rank,
    plot_slice,
    plot_timeline,
)

from logging_config import logger
from utils import get_db_url


# Function to load study from URL
def load_study_from_url(study_name, url):
    storage = optuna.storages.RDBStorage(url)
    return optuna.load_study(study_name=study_name, storage=storage)


# Function to plot and save optimization history
def plot_and_save_optimization_history(study, output_dir):
    fig = plot_optimization_history(study)
    fig.write_image(os.path.join(output_dir, "optimization_history.png"))


# Function to plot and save parameter importances
def plot_and_save_param_importances(study, output_dir):
    fig = plot_param_importances(study)
    fig.write_image(os.path.join(output_dir, "param_importances.png"))


# Function to plot and save intermediate values
def plot_and_save_intermediate_values(study, output_dir):
    fig = plot_intermediate_values(study)
    fig.write_image(os.path.join(output_dir, "intermediate_values.png"))


# Function to plot and save parallel coordinates
def plot_and_save_parallel_coordinate(study, output_dir):
    fig = plot_parallel_coordinate(study)
    fig.write_image(os.path.join(output_dir, "parallel_coordinate.png"))


# Function to plot and save contour
def plot_and_save_contour(study, output_dir):
    fig = plot_contour(study)
    fig.write_image(os.path.join(output_dir, "contour.png"))


# Function to plot and save slice
def plot_and_save_slice(study, output_dir):
    fig = plot_slice(study)
    fig.write_image(os.path.join(output_dir, "slice.png"))


# Function to plot and save EDF
def plot_and_save_edf(study, output_dir):
    fig = plot_edf(study)
    fig.write_image(os.path.join(output_dir, "edf.png"))


# Function to plot and save rank
def plot_and_save_rank(study, output_dir):
    fig = plot_rank(study)
    fig.write_image(os.path.join(output_dir, "rank.png"))


# Function to plot and save timeline
def plot_and_save_timeline(study, output_dir):
    fig = plot_timeline(study)
    fig.write_image(os.path.join(output_dir, "timeline.png"))


# Main analysis function
def analyze_study(study_name, study_url, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    study = load_study_from_url(study_name, study_url)

    plot_and_save_optimization_history(study, output_dir)
    plot_and_save_param_importances(study, output_dir)
    plot_and_save_intermediate_values(study, output_dir)
    plot_and_save_parallel_coordinate(study, output_dir)
    plot_and_save_contour(study, output_dir)
    plot_and_save_slice(study, output_dir)
    plot_and_save_edf(study, output_dir)
    plot_and_save_rank(study, output_dir)
    plot_and_save_timeline(study, output_dir)
    best_trial = study.best_trial
    print("Best trial:")
    print(f"  Value: {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    load_dotenv()
    logger.info("Starting analysis...")
    study_name = '1M_steps'
    study_url = get_db_url()
    output_dir = os.getenv('OUTPUTS_DIR', 'outputs')

    analyze_study(study_name, study_url, output_dir)

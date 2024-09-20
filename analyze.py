import os

import matplotlib.pyplot as plt
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances


# Function to load study from URL
def load_study_from_url(study_name, url):
    storage = optuna.storages.RDBStorage(
        url,
        engine_kwargs={"connect_args": {"check_same_thread": False}}
    )
    return optuna.load_study(study_name=study_name, storage=storage)

# Function to plot and save parameter importances
def plot_and_save_param_importances(study, output_dir):
    fig = plot_param_importances(study)
    fig.write_image("param_importances.png")
    plt.title("Parameter Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "param_importances.png"))
    plt.close()

# Function to plot and save optimization history
def plot_and_save_optimization_history(study, output_dir):
    fig = plot_optimization_history(study)
    fig.write_image("optimization_history.png")
    plt.title("Optimization History")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimization_history.png"))
    plt.close()

# Main analysis function
def analyze_study(study_name, study_url, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load study from URL
    study = load_study_from_url(study_name, study_url)

    # Plot and save parameter importances
    plot_and_save_param_importances(study, output_dir)

    # Plot and save optimization history
    plot_and_save_optimization_history(study, output_dir)

    # Print best trial information
    best_trial = study.best_trial
    print("Best trial:")
    print(f"  Value: {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    # Replace with your actual study URL
    study_name = '1M_steps'
    study_url = "sqlite:///path/to/your/study.db"
    output_dir = "outputs"

    analyze_study(study_name, study_url, output_dir)

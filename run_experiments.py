from optimize import run_optimization
from train import train_model
import optuna

def run_experiment(experiment_config):
    study_name = experiment_config['study_name']
    n_trials = experiment_config['n_trials']
    env_name = experiment_config['env_name']
    total_timesteps = experiment_config['total_timesteps']
    param_ranges = experiment_config['param_ranges']
    direction = experiment_config.get('direction', 'maximize')
    storage = experiment_config.get('storage', None)
    sampler = experiment_config.get('sampler', None)
    pruner = experiment_config.get('pruner', None)

    # Run the optimization study
    study = run_optimization(
        study_name=study_name,
        n_trials=n_trials,
        env_name=env_name,
        total_timesteps=total_timesteps,
        storage=storage,
    )

    # Train a final model with the best parameters
    best_params = study.best_params
    final_mean_reward, final_std_reward = train_model(
        best_params,
        env_name=env_name,
        total_timesteps=total_timesteps * 10,  # Train for longer with best params
        save_path=f"best_model_{study_name}"
    )
    print(f"Final model performance for {study_name}: {final_mean_reward:.2f} +/- {final_std_reward:.2f}")

    return study, final_mean_reward, final_std_reward

if __name__ == "__main__":
    # Define different experiment configurations
    common_param_ranges = {
        'learning_rate': {'type': 'loguniform', 'low': 1e-5, 'high': 1},
        'n_steps': {'type': 'int', 'low': 16, 'high': 2048},
        'gamma': {'type': 'uniform', 'low': 0.9, 'high': 0.9999},
        'gae_lambda': {'type': 'uniform', 'low': 0.9, 'high': 1.0},
        'ent_coef': {'type': 'loguniform', 'low': 1e-8, 'high': 1e-1},
        'activation_fn': {'type': 'categorical', 'choices': ['tanh', 'relu']}
    }

    experiments = [
        {
            'study_name': 'lunar_lander_short',
            'n_trials': 50,
            'env_name': 'LunarLander-v2',
            'total_timesteps': 100000,
            'param_ranges': common_param_ranges,
            'direction': 'maximize',
            'storage': 'sqlite:///lunar_lander_short.db',
            'sampler': optuna.samplers.TPESampler(),
            'pruner': optuna.pruners.MedianPruner()
        },
        {
            'study_name': 'lunar_lander_long',
            'n_trials': 100,
            'env_name': 'LunarLander-v2',
            'total_timesteps': 500000,
            'param_ranges': common_param_ranges,
            'direction': 'maximize',
            'storage': 'sqlite:///lunar_lander_long.db',
            'sampler': optuna.samplers.CmaEsSampler(),
            'pruner': optuna.pruners.HyperbandPruner()
        },
        # Add more experiment configurations as needed
    ]

    # Run all experiments
    for experiment_config in experiments:
        study, mean_reward, std_reward = run_experiment(experiment_config)
        print(f"Experiment {experiment_config['study_name']} completed.")
        print(f"Best performance: {mean_reward:.2f} +/- {std_reward:.2f}")
        print("--------------------")

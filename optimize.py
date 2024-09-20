import optuna
from train import train_model

def create_objective(env_name, total_timesteps):
    def objective(trial):
        # Define the hyperparameters to optimize
        params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1),
            'n_steps': trial.suggest_int('n_steps', 16, 2048),
            'gamma': trial.suggest_uniform('gamma', 0.9, 0.9999),
            'gae_lambda': trial.suggest_uniform('gae_lambda', 0.9, 1.0),
            'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        }

        # Train the model with these parameters
        mean_reward, _ = train_model(params, env_name=env_name, total_timesteps=total_timesteps)

        return mean_reward

    return objective

def run_optimization(study_name, n_trials, env_name, total_timesteps, storage=None):
    if storage:
        study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage, load_if_exists=True)
    else:
        study = optuna.create_study(direction='maximize', study_name=study_name)

    objective = create_objective(env_name, total_timesteps)
    study.optimize(objective, n_trials=n_trials)

    print(f'Best trial for study {study_name}:')
    trial = study.best_trial

    print('Value: ', trial.value)
    print('Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    return study

if __name__ == "__main__":
    # This can be used for quick testing
    study = run_optimization("test_study", n_trials=10, env_name="LunarLander-v2", total_timesteps=100000)

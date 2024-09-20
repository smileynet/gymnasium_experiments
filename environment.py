# In environment.py

import os
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

LOG_DIR = os.getenv("LOG_DIR", "./logs")


def create_env(env_id, n_envs=1):
    if n_envs == 1:
        env = gym.make(env_id)
        env = DummyVecEnv([lambda: env])
    else:
        env = make_vec_env(
            env_id,
            n_envs=n_envs,
            monitor_dir=os.path.join(LOG_DIR, "train"),
            seed=0,
            vec_env_cls=None,
        )
    return VecMonitor(env)


def create_eval_env(env_id):
    env = gym.make(env_id)
    env = DummyVecEnv([lambda: env])
    return VecMonitor(env, os.path.join(LOG_DIR, "eval"))

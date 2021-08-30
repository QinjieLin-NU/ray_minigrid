import ray
from ray import tune
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
import gym
import gym_minigrid
from train_ray import MyEnv,load_config,RegisterEnv
import numpy as np
import pickle
from ray.tune.suggest.bayesopt import BayesOptSearch

#register env
def env_creator(env_config):
    from train_ray import RegisterEnv
    return RegisterEnv('MiniGrid-KeyCorridorS3R3-v0')   # return an env instance
register_env("my_env", env_creator)

def on_train_result(info):
    result = info["result"]
    trainer = info["trainer"]
    print("training")

ray.init(num_cpus=30)
tune.run(
    "PPO",
    stop={"training_iteration": 10000, "episode_reward_mean": 0.95},
    config={
        "env": "my_env",
        "framework": "torch",
        "lr":  0.0003,
        "train_batch_size": tune.grid_search([160000, 80000]),
        "sgd_minibatch_size": tune.grid_search([3200, 1600]) ,
        "num_sgd_iter" : tune.grid_search([2,20]),
        "gamma": tune.grid_search([0.995,0.9]),
        "num_workers": 2,
        "num_envs_per_worker": 5,
        "observation_filter": "MeanStdFilter",
        "model": {
            "conv_filters": [
                [32, [2, 2], 1],
                [32, [2, 2], 2],
                [32, [2, 2], 2],
                [32, [2, 2], 2],
            ]
        },
        "callbacks": {
            "on_train_result": on_train_result,
        },
    },
    # resources_per_trial=tune.PlacementGroupFactory(
        # [{"CPU": 2}] 
    # ),
)
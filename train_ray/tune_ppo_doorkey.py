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

env_name = 'MiniGrid-DoorKey-MediumRawPartial-v0' #'MiniGrid-DoorKey-MediumRaw-v0'
ray_envname = 'Ray-%s'%env_name
#register env
def env_creator(env_config):
    from train_ray import RegisterEnv
    return RegisterEnv(env_name)   # return an env instance
register_env(ray_envname, env_creator)

def on_train_result(info):
    result = info["result"]
    trainer = info["trainer"]
    print("training")

ray.init(num_cpus=80)
tune.run(
    "PPO",
    stop={"training_iteration": 500, "episode_reward_mean": 0.95},
    config={
        "env": ray_envname,
        "framework": "torch",
        "lr": tune.grid_search([3e-4,5e-3,5e-5]), #0.001 0.0003
        "seed": tune.grid_search([12345, 45678]),
        "train_batch_size": tune.grid_search([64000]), #640000 6400
        "sgd_minibatch_size": tune.grid_search([256,512,1024]) , #1024
        "num_sgd_iter" : 2,
        "gamma": 0.995,
        "rollout_fragment_length":  tune.grid_search([500]),
        "num_workers": 5, #5
        "num_envs_per_worker": 5, #1
        # "observation_filter": "MeanStdFilter",
        "model": {
            # "use_lstm":  tune.grid_search([False]),
            # "conv_activation": "relu",
            "dim": 9,
            # "grayscale": tune.grid_search([True]),
            "conv_filters": [
                [32, [3, 3], 1],
                [32, [2, 2], 2],
                [32, [2, 2], 2],
                [32, [2, 2], 2],
            ],
            # "fcnet_hiddens": [256, 256],
            # "fcnet_activation": "relu",
        },
        "callbacks": {
            "on_train_result": on_train_result,
        },
    },
    # resources_per_trial=tune.PlacementGroupFactory(
        # [{"CPU": 2}] 
    # ),
)
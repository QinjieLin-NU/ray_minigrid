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
    return RegisterEnv('MiniGrid-DoorKey-8x8-v0')   # return an env instance
register_env("my_env", env_creator)

def on_train_result(info):
    result = info["result"]
    trainer = info["trainer"]
    print("training")

ray.init(num_cpus=20)
tune.run(
    "PPO",
    stop={"training_iteration": 1000, "episode_reward_mean": 0.95},
    config={
        "env": "my_env",
        "framework": "torch",
        "lr":  0.0003,
        "seed": 123,
        "train_batch_size": 3200,
        "sgd_minibatch_size": 256 ,
        "num_sgd_iter" :2,
        "num_workers": 2,
        # "observation_filter": "MeanStdFilter",
        "model": {
            "use_lstm": tune.grid_search([True, False]),
            "conv_activation": "relu",
            "dim": 7,
            "grayscale": True,
            # zero_mean: false
            # Reduced channel depth and kernel size from default
            # fcnet_hiddens: [256, 256]
            # fcnet_activation: relu
            "conv_filters": [
                [32, [7, 7], 3],
                [32, [3, 3], 3],
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
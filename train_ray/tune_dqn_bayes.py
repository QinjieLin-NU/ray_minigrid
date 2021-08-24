import ray
from ray import tune
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.agents.dqn import DQNTrainer
import ray.rllib.agents.dqn as dqn
import gym
import gym_minigrid
from train_ray import MyEnv,load_config,RegisterEnv
import numpy as np
import pickle
from ray.tune.suggest.bayesopt import BayesOptSearch

#register env
def env_creator(env_config):
    from train_ray import RegisterEnv
    return RegisterEnv('MiniGrid-DoorKey-5x5-v0')   # return an env instance
register_env("my_env", env_creator)

def on_train_result(info):
    result = info["result"]
    trainer = info["trainer"]
    tune.report(my_loss=intermediate_score)

ray.init(num_cpus=20)
bayesopt = BayesOptSearch(metric="my_loss", mode="min")
tune.run(
    "DQN",
    config={
        "env": "my_env",
        "framework": "torch",
        "lr":  tune.uniform(10,20),
        "train_batch_size": tune.uniform(32,256),
        "num_workers": 2,
        "model": {
            "conv_filters": [
            [32, [7, 7], 3],
            [32, [3, 3], 3],
            ]
        },
        "callbacks": {
            "on_train_result": on_train_result,
        },
    },
    search_alg=bayesopt
    # resources_per_trial=tune.PlacementGroupFactory(
        # [{"CPU": 2}] 
    # ),
)
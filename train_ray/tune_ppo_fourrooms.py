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

env_name = 'MiniGrid-FourRooms-Continuous-v0'
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

ray.init(num_cpus=30)
tune.run(
    "PPO",
    stop={"training_iteration": 1000, "episode_reward_mean": 0.95},
    config={
        "env": ray_envname,
        "framework": "torch",
        "lr":  0.0003,
        "train_batch_size": tune.grid_search([2560,40000]),
        "sgd_minibatch_size": tune.grid_search([256,512,1024]) ,
        "num_sgd_iter" : tune.grid_search([2,10]),
        "gamma": 0.995,
        "rollout_fragment_length":  500,
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
        "seed":12345,
        "callbacks": {
            "on_train_result": on_train_result,
        },
    },
    # resources_per_trial=tune.PlacementGroupFactory(
        # [{"CPU": 2}] 
    # ),
)
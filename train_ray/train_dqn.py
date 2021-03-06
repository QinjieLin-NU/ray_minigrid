import ray
from ray import tune
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.agents.dqn import DQNTrainer
import ray.rllib.agents.dqn as dqn
import gym
import gym_minigrid
from train_ray import MyEnv,load_config
import numpy as np
import pickle

import warnings
warnings.filterwarnings("ignore")

import uuid
uni_str = uuid.uuid4().hex[:8].upper()
ray.init(num_cpus=20,log_to_driver=False)
config = dqn.DEFAULT_CONFIG.copy()
# loaded_config = load_config("train_ray/config_dqn.yaml")
loaded_config = load_config("train_ray/config_dqn_fourrooms.yaml")
loaded_config["output"] = "%s_%s"%(loaded_config["output"],uni_str)
for key, value in loaded_config.items():
    config[key] = value
    print(key,value)

trainer = dqn.DQNTrainer(config=config, env=MyEnv)

# Perform iterations of training the policy with PPO
reward_array = []
for i in range(1500):
    result = trainer.train()
    mean_reward = result["episode_reward_mean"]
    reward_array.append(mean_reward)
    print("iteration",i," reward_mean: ",mean_reward)
    if i % 25 == 0:
        print("saving training trajectoires at",loaded_config["output"])
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)

#save learning curve
with open('data/dqn.pkl', 'wb') as f:
    pickle.dump(reward_array, f)

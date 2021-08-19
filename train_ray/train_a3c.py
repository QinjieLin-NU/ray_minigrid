import ray
from ray import tune
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.agents.a3c import A3CTrainer
import ray.rllib.agents.a3c as a3c
import gym
import gym_minigrid
from train_ray import MyEnv,load_config
import numpy as np
import pickle

#trainer config, refer config to ppo tuned example
ray.init()
config = a3c.DEFAULT_CONFIG.copy()
loaded_config = load_config("train_ray/config_a3c.yaml")
for key, value in loaded_config.items():
    config[key] = value
    print(key,value)

trainer = a3c.A3CTrainer(config=config, env=MyEnv)

# Perform iterations of training the policy with PPO
reward_array = []
for i in range(500):
    result = trainer.train()
    mean_reward = result["episode_reward_mean"]
    reward_array.append(mean_reward)
    print("iteration",i," reward_mean: ",mean_reward)
    if i % 25 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)

#save learning curve
with open('data/a3c.pkl', 'wb') as f:
    pickle.dump(reward_array, f)

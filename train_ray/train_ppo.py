import ray
from ray import tune
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.agents.ppo import PPOTrainer
import ray.rllib.agents.ppo as ppo
import gym
import gym_minigrid
from train_ray import MyEnv,load_config
import numpy as np
import pickle

#trainer config, refer config to ppo tuned example
ray.init(num_cpus=20)
config = ppo.DEFAULT_CONFIG.copy()
loaded_config = load_config("train_ray/config_ppo_corridor.yaml")
for key, value in loaded_config.items():
    config[key] = value
    print(key,value)
trainer = ppo.PPOTrainer(config=config, env=MyEnv)

# Perform iterations of training the policy with PPO
reward_array = []
for i in range(1000):
    result = trainer.train()
    mean_reward = result["episode_reward_mean"]
    reward_array.append(mean_reward)
    print("iteration",i," reward_mean: ",mean_reward)
    if i % 25 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)

#save learning curve
with open('data/ppo.pkl', 'wb') as f:
    pickle.dump(reward_array, f)

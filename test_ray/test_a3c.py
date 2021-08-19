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
import time
from ray.rllib.models.preprocessors import get_preprocessor

#trainer config
ray.init()
config = a3c.DEFAULT_CONFIG.copy()
loaded_config = load_config("train_ray/config_a3c.yaml")
for key, value in loaded_config.items():
    config[key] = value
    print(key,value)
trainer = a3c.A3CTrainer(config=config, env=MyEnv)
restore_path = "/root/ray_results/A3C_MyEnv_2021-08-19_04-29-27z0q2g459/checkpoint_000476/checkpoint-476" # 
trainer.restore(restore_path) 

# run until episode ends
env = MyEnv(loaded_config['env_config'])
env.render('human')
for i in range(100):
    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = trainer.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        
        env.render('human')
        time.sleep(0.1)
    print("episode reward:",episode_reward)
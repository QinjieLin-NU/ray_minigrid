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
import time

#trainer config
ray.init()
config = ppo.DEFAULT_CONFIG.copy()
loaded_config = load_config("test_ray/test_ppo.yaml")
for key, value in loaded_config.items():
    config[key] = value
    print(key,value)
trainer = ppo.PPOTrainer(config=config, env=MyEnv)
restore_path = "/root/ray_results/PPO_MyEnv_2021-08-19_04-35-49rg4jc1k5/checkpoint_000476/checkpoint-476" # 
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
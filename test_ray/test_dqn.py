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
import time

#trainer config
ray.init()
config = dqn.DEFAULT_CONFIG.copy()
loaded_config = load_config("test_ray/config/test_dqn.yaml")
for key, value in loaded_config['trainer_config'].items():
    config[key] = value
    print(key,value)
trainer = dqn.DQNTrainer(config=config, env=MyEnv)
restore_path = loaded_config["test_config"]["restore_path"]
trainer.restore(restore_path) 

# run until episode ends
env = MyEnv(loaded_config['trainer_config']['env_config'])
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
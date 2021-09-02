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

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--config', type=str, default="test_ray/config/test_ppo_corridor.yaml",
                    help='')
args = parser.parse_args()
config_file = args.config
print(config_file)

#trainer config
ray.init()
config = ppo.DEFAULT_CONFIG.copy()
loaded_config = load_config(config_file)
for key, value in loaded_config['trainer_config'].items():
    config[key] = value
    print(key,value)
trainer = ppo.PPOTrainer(config=config, env=MyEnv)
restore_path = loaded_config["test_config"]["restore_path"]
trainer.restore(restore_path) 

# run until episode ends
state = trainer.get_policy().model.get_initial_state()
env = MyEnv(loaded_config['trainer_config']['env_config'])
env.render('human')
env.seed(0)
for i in range(100):
    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        # action, state, logit = trainer.compute_action(observation=obs, prev_action=1.0, 
                                        # prev_reward = 0.0, state = state)
        action = trainer.compute_action(observation=obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        
        env.render('human')
        # time.sleep(0.1)
    print("episode reward:",episode_reward)
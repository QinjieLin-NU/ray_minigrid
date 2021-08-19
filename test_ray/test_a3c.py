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

#trainer config
ray.init()
config = a3c.DEFAULT_CONFIG.copy()
loaded_config = load_config("test_ray/config/test_a3c.yaml")
for key, value in loaded_config["trainer_config"].items():
    config[key] = value
    print(key,value)
trainer = a3c.A3CTrainer(config=config, env=MyEnv)
restore_path = loaded_config["test_config"]["restore_path"]
trainer.restore(restore_path) 

# run until episode ends
state = trainer.get_policy().model.get_initial_state()
env = MyEnv(loaded_config['trainer_config']['env_config'])
env.render('human')
for i in range(100):
    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action, state, logit = trainer.compute_action(observation=obs, prev_action=1.0, 
                                        prev_reward = 0.0, state = state)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        
        env.render('human')
        time.sleep(0.1)
    print("episode reward:",episode_reward)
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
import envs
import uuid
uni_str = uuid.uuid4().hex[:8].upper()

#trainer config, refer config to ppo tuned example
ray.init(num_cpus=20)
config = ppo.DEFAULT_CONFIG.copy()
loaded_config = load_config("train_ray/config_ppo_fourrooms_notFilter.yaml")
loaded_config["output"] = "%s_%s"%(loaded_config["output"],uni_str)
for key, value in loaded_config.items():
    config[key] = value
    print(key,value)
trainer = ppo.PPOTrainer(config=config, env=MyEnv)

# Perform iterations of training the policy with PPO
reward_array = []
file_name = f"data/ppo_{config['env_config']['env_name']}_{config['gamma']}_{config['num_sgd_iter']}_{config['sgd_minibatch_size']}_{config['train_batch_size']}.pkl"
for i in range(200): #1000
    result = trainer.train()
    mean_reward = result["episode_reward_mean"]
    reward_array.append(mean_reward)
    print("iteration",i," reward_mean: ",mean_reward)
    if (i+1) % 25 ==0 or i==0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
        print("saving training trajectoires at",loaded_config["output"])
        #save learning curve
        with open(file_name, 'wb') as f:
            pickle.dump(reward_array, f)
#save learning curve
with open(file_name, 'wb') as f:
    pickle.dump(reward_array, f)

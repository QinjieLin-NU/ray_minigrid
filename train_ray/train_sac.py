import ray
from ray import tune
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.agents.sac import SACTrainer
import ray.rllib.agents.sac as sac
import gym
import gym_minigrid
from train_ray import MyEnv,load_config
import numpy as np
import pickle
import envs
import uuid
uni_str = uuid.uuid4().hex[:8].upper()

ray.init(num_cpus=20)
config = sac.DEFAULT_CONFIG.copy()
loaded_config = load_config("train_ray/config_sac.yaml")
loaded_config["output"] = "%s_%s"%(loaded_config["output"],uni_str)
for key, value in loaded_config.items():
    config[key] = value
    print(key,value)
trainer = sac.SACTrainer(config=config, env=MyEnv)

reward_array = []
file_name = f"data/sac_{config['env_config']['env_name']}.pkl"
for i in range(1500): #200 1000
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

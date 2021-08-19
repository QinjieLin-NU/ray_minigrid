import yaml
import gym
import gym_minigrid

def load_config(file_path):
    with open(file_path) as f:
        ppo_config = yaml.safe_load(f)
    return ppo_config

#register env
# def env_creator(env_config):
#     import gym_minigrid
#     from gym_minigrid.wrappers import RGBImgPartialObsWrapper,ImgObsWrapper
#     env = gym.make('MiniGrid-Empty-5x5-v0') 
#     env = RGBImgPartialObsWrapper(env)
#     env = ImgObsWrapper(env)
#     return env 
# register_env("my_env", env_creator)
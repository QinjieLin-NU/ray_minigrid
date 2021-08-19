import gym
import gym_minigrid
from gym_minigrid.wrappers import RGBImgPartialObsWrapper,ImgObsWrapper
env = gym.make('MiniGrid-GoToDoor-8x8-v0') 
# env = RGBImgPartialObsWrapper(env)
# env = ImgObsWrapper(env)
obs = env.reset()
print(obs["image"].shape)
print(env.observation_space['image'])
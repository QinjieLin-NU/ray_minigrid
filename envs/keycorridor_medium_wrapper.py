import gym
import gym_minigrid
from gym_minigrid.wrappers import FullyObsWrapper

import numpy as np
class KeyCorridorMediumWrapper(gym.Env):
    def __init__(self):
        self.env = gym.make('MiniGrid-KeyCorridorS4R3-v0')  

        self.env.unwrapped.agent_view_size=7
        self.env.unwrapped.see_through_walls=False
        self.env.unwrapped.width= 10
        self.env.unwrapped.height = 10 
        # self.env.unwrapped.grid_size = 1

        self.env = FullyObsWrapper(self.env)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.actions = self.env.actions
        self.step_count = self.env.step_count


    def reset(self):
        _state = self.env.reset()
        self.step_count = self.env.step_count
        return _state

    def step(self, action):
        _state, _reward, _d, _info = self.env.step(action)
        self.step_count = self.env.step_count
        return _state, _reward, _d, _info

    def render(self,mode,**args):
        _return = self.env.render(mode,**args)
        return _return
    
    def seed(self,seed=None):
        self.env.seed(seed)
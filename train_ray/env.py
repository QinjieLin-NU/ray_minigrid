import gym
import gym_minigrid
import envs

class MyEnv(gym.Env):
    def __init__(self, env_config):
        self.env = gym.make(env_config['env_name']) 
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space['image']

    def reset(self):
        return self.env.reset()['image']

    def step(self, action):
        _state, _r, _d, _info = self.env.step(action)
        _obs = _state['image']
        return _obs, _r, _d, _info

    def render(self,mode):
        self.env.render(mode)
        return

class RegisterEnv(gym.Env):
    def __init__(self, env_name):
        self.env = gym.make(env_name) 
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space['image']

    def reset(self):
        return self.env.reset()['image']

    def step(self, action):
        _state, _r, _d, _info = self.env.step(action)
        _obs = _state['image']
        return _obs, _r, _d, _info

    def render(self,mode):
        self.env.render(mode)
        return
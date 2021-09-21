import gym
import gym_minigrid

import numpy as np
class FourRoomContinuous(gym.Env):
    def __init__(self):
        self.env = gym.make("MiniGrid-FourRooms-v0") 
        # self.env = DirectionObsWrapper(self.env)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.actions = self.env.actions
        self.step_count = self.env.step_count

    def calc_distance(self,pos,goal):
        mahattan_distance = np.abs(pos[0]-goal[0]) + np.abs(pos[1]-goal[1])
        return mahattan_distance
    
    def get_goal(self):
        i = 0
        goal_x,goal_y = 0,0
        for grid in self.env.grid.grid:
            if grid is not None and grid.type == "goal":
                goal_x,goal_y =  i%self.env.grid.width , i//self.env.grid.width
            i+=1
        return [goal_x,goal_y]

    def reset(self):
        _obs = self.env.reset()
        self.cur_goal = self.get_goal()
        self.last_distance = self.calc_distance(self.env.agent_pos,self.cur_goal)
        return _obs

    def step(self, action):
        _state, _reward, _d, _info = self.env.step(action)
        self.step_count = self.env.step_count
        _obs = _state

        #calac reward
        current_distannce = self.calc_distance(self.env.agent_pos,self.cur_goal)
        if _d:
            if _reward > 0:
                _reward = max(self.env.width, self.env.height) * _reward
        else:
            _reward = self.last_distance - current_distannce
        self.last_distance  = current_distannce

        return _obs, _reward, _d, _info

    def render(self,mode,**args):
        _return = self.env.render(mode,**args)
        return _return
    
    def seed(self,seed=None):
        self.env.seed(seed)
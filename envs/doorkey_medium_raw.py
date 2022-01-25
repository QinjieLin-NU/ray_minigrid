import gym
import gym_minigrid
from gym import spaces
import numpy as np

from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class DoorKeyCustomEnv(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, size=8):
        super().__init__(
            grid_size=size,
            max_steps=10*size*size
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width-2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width-2)
        self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        self.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(splitIdx, height)
        )

        self.mission = "use the key to open the door and then get to the goal"

class DoorKeyMediumRaw(gym.Env):
    def __init__(self):
        self.env = DoorKeyCustomEnv() #gym.make('MiniGrid-DoorKey-8x8-v0')  
        self.env.width= 10
        self.env.height = 10 
        self.tile_size = 1

        # self.env = DirectionObsWrapper(self.env)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width*self.tile_size, self.env.height*self.tile_size, 3),  # number of cells
            dtype='uint8'
        )
        self.actions = self.env.actions
        self.step_count = self.env.step_count


    def reset(self):
        _state = self.env.reset()
        self.step_count = self.env.step_count
        _render_img = self.env.render('rgb_array', tile_size=self.tile_size )
        _state['image'] = _render_img
        return _state

    def step(self, action):
        _state, _reward, _d, _info = self.env.step(action)
        self.step_count = self.env.step_count
        _render_img = self.env.render('rgb_array', tile_size=self.tile_size )
        _state['image'] = _render_img
        return _state, _reward, _d, _info

    def render(self,mode,**args):
        _return = self.env.render(mode,**args)
        return _return
    
    def seed(self,seed=None):
        self.env.seed(seed)
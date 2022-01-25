import gym
import gym_minigrid
from gym import spaces
import numpy as np
from gym_minigrid.minigrid import MiniGridEnv
from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class FourRoomsCustomEnv(MiniGridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, agent_pos=None, goal_pos=None, input_maxstep=None):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        super().__init__(width=11,height=11, max_steps=input_maxstep, agent_view_size=9,see_through_walls=True,)

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    tmp_yT,tmp_yB = (yT+1, yB) if yT==0 else (yT, yB-1) #qinjie
                    pos = (xR, self._rand_int(tmp_yT,tmp_yB)) #qinjie
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    tmp_xL,tmp_xR = (xL+1, xR) if xL==0 else (xL, xR-1)  #qinjie
                    pos = (self._rand_int(tmp_xL,tmp_xR), yB) #qinjie
                    self.grid.set(*pos, None)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())

        self.mission = 'Reach the goal'

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        return obs, reward, done, info


class FourroomsMediumRawPartial(gym.Env):
    def __init__(self,input_maxstep):
        self.input_maxstep =input_maxstep
        self.env = FourRoomsCustomEnv(input_maxstep=input_maxstep)
        self.tile_size = 1

        # self.env = DirectionObsWrapper(self.env)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        # self.observation_space.spaces["image"] = spaces.Box(
        #     low=0,
        #     high=255,
        #     shape=(self.env.width*self.tile_size, self.env.height*self.tile_size, 3),  # number of cells
        #     dtype='uint8'
        # )
        self.actions = self.env.actions
        self.step_count = self.env.step_count


    def reset(self):
        _state = self.env.reset()
        self.step_count = self.env.step_count
        # _render_img = self.env.render('rgb_array', tile_size=self.tile_size )
        # _state['image'] = _render_img

        self.last_state = _state
        return _state

    def step(self, action):
        _state, _reward, _d, _info = self.env.step(action)
        self.step_count = self.env.step_count
        # _render_img = self.env.render('rgb_array', tile_size=self.tile_size )
        # _state['image'] = _render_img

        self.last_state = _state
        return _state, _reward, _d, _info

    def render(self,mode,**args):
        _return = self.env.render(mode,**args)
        return _return
    
    def seed(self,seed=None):
        self.env.seed(seed)
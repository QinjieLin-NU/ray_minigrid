import gym
import gym_minigrid
from gym import spaces
import numpy as np

from gym_minigrid.roomgrid import RoomGrid
from gym_minigrid.register import register

class KeyCorridorCustom(RoomGrid):
    """
    A ball is behind a locked door, the key is placed in a
    random room.
    """

    def __init__(
        self,
        num_rows=3,
        obj_type="ball",
        room_size=6,
        seed=None,
        input_maxstep=None,
    ):
        self.obj_type = obj_type

        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            max_steps=input_maxstep, #30*room_size**2,
            seed=seed,
            agent_view_size=9
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Connect the middle column rooms into a hallway
        for j in range(1, self.num_rows):
            self.remove_wall(1, j, 3)

        # Add a locked door on the bottom right
        # Add an object behind the locked door
        room_idx = self._rand_int(0, self.num_rows)
        door, _ = self.add_door(2, room_idx, 2, locked=True)
        obj, _ = self.add_object(2, room_idx, kind=self.obj_type)

        # Add a key in a random room on the left side
        self.add_object(0, self._rand_int(0, self.num_rows), 'key', door.color)

        # Place the agent in the middle
        self.place_agent(1, self.num_rows // 2)

        # Make sure all rooms are accessible
        self.connect_all()

        self.obj = obj
        self.mission = "pick up the %s %s" % (obj.color, obj.type)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.obj:
                reward = self._reward()
                done = True

        return obs, reward, done, info


class KeyCorridorS4R3Custom(KeyCorridorCustom):
    def __init__(self, input_maxstep=None, seed=None):
        super().__init__(
            room_size=4,
            num_rows=3,
            seed=seed,
            input_maxstep=input_maxstep,
        )

class KeyCorridorMediumRawPartial(gym.Env):
    def __init__(self,input_maxstep):
        self.input_maxstep = input_maxstep
        self.env = KeyCorridorS4R3Custom(input_maxstep=input_maxstep) 

        self.env.see_through_walls=True
        # self.env.unwrapped.width= 10 # it happens to be 10, width and heitgh is specified in Rooomgrid
        # self.env.unwrapped.height = 10 
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
        return _state

    def step(self, action):
        _state, _reward, _d, _info = self.env.step(action)
        self.step_count = self.env.step_count
        # _render_img = self.env.render('rgb_array', tile_size=self.tile_size )
        # _state['image'] = _render_img
        return _state, _reward, _d, _info

    def render(self,mode,**args):
        _return = self.env.render(mode,**args)
        return _return
    
    def seed(self,seed=None):
        self.env.seed(seed)
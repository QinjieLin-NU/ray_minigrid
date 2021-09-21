from gym.envs.registration import register
import gym_minigrid
from gym_minigrid.wrappers import DirectionObsWrapper
# this is the base implementation od environment
register(
	id='MiniGrid-FourRooms-Continuous-v0',
	entry_point='envs.fourrooms_continuous:FourRoomContinuous',
	max_episode_steps=100,
	reward_threshold=100.0,
	)
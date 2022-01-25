from gym.envs.registration import register
import gym_minigrid
from gym_minigrid.wrappers import DirectionObsWrapper
# this is the base implementation od environment
fourrooms_max_step = 100 #
corridormax_step =  640 # 240
doorkey_max_step = 640 #

register(
	id='MiniGrid-FourRooms-Continuous-v0',
	entry_point='envs.fourrooms_continuous:FourRoomContinuous',
	max_episode_steps=100,
	reward_threshold=100.0,
	)

# ================fourrooms=================
register(
	id='MiniGrid-FourRooms-Medium-v0',
	entry_point='envs.fourrooms_medium:FourroomsMedium',
	max_episode_steps=fourrooms_max_step,
	reward_threshold=100.0,
	)
register(
	id='MiniGrid-FourRooms-MediumWrapper-v0',
	entry_point='envs.fourrooms_medium_wrapper:FourroomsMediumWrapper',
	max_episode_steps=fourrooms_max_step,
	reward_threshold=100.0,
	)
register(
	id='MiniGrid-FourRooms-MediumRaw-v0',
	entry_point='envs.fourrooms_medium_raw:FourroomsMediumRaw',
	max_episode_steps=fourrooms_max_step,
	reward_threshold=100.0,
	)
register(
	id='MiniGrid-FourRooms-MediumRawPartial-v0',
	entry_point='envs.fourrooms_medium_raw_partial:FourroomsMediumRawPartial',
	max_episode_steps=fourrooms_max_step,
	reward_threshold=100.0,
	kwargs={'input_maxstep': fourrooms_max_step},
	)

# ================corridor=================
register(
	id='MiniGrid-KeyCorridor-Medium-v0',
	entry_point='envs.keycorridor_medium:KeyCorridorMedium',
	max_episode_steps=corridormax_step,
	reward_threshold=100.0,
	)
register(
	id='MiniGrid-KeyCorridor-MediumWrapper-v0',
	entry_point='envs.keycorridor_medium_wrapper:KeyCorridorMediumWrapper',
	max_episode_steps=corridormax_step,
	reward_threshold=100.0,
	)
register(
	id='MiniGrid-KeyCorridor-MediumRaw-v0',
	entry_point='envs.keycorridor_medium_raw:KeyCorridorMediumRaw',
	max_episode_steps=corridormax_step,
	reward_threshold=100.0,
	)
register(
	id='MiniGrid-KeyCorridor-MediumRawPartial-v0',
	entry_point='envs.keycorridor_medium_raw_partial:KeyCorridorMediumRawPartial',
	max_episode_steps=corridormax_step,
	reward_threshold=100.0,
	kwargs={'input_maxstep': corridormax_step},
	)
# ================doorkey=================
register(
	id='MiniGrid-DoorKey-Medium-v0',
	entry_point='envs.doorkey_medium:DoorKeyMedium',
	max_episode_steps=doorkey_max_step,
	reward_threshold=100.0,
	)
register(
	id='MiniGrid-DoorKey-MediumWrapper-v0',
	entry_point='envs.doorkey_medium_wrapper:DoorKeyMediumWrapper',
	max_episode_steps=doorkey_max_step,
	reward_threshold=100.0,
	)
register(
	id='MiniGrid-DoorKey-MediumRaw-v0',
	entry_point='envs.doorkey_medium_raw:DoorKeyMediumRaw',
	max_episode_steps=doorkey_max_step,
	reward_threshold=100.0,
	)
register(
	id='MiniGrid-DoorKey-MediumRawPartial-v0',
	entry_point='envs.doorkey_medium_raw_partial:DoorKeyMediumRawPartial',
	max_episode_steps=doorkey_max_step,
	reward_threshold=100.0,
	kwargs={'input_maxstep': doorkey_max_step},
	)
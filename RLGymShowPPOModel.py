import rlgym
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition,GoalScoredCondition
from rlgym.utils.state_setters import RandomState, DefaultState
from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.obs_builders import ObsBuilder, AdvancedObs
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.reward_functions.common_rewards import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.gamestates import GameState

from rlgym_tools.extra_obs.advanced_padder import AdvancedObsPadder
from rlgym_tools.extra_rewards.kickoff_reward import KickoffReward
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter
from rlgym_tools.extra_state_setters.goalie_state import GoaliePracticeState

from rlgym_ppo.ppo import MultiDiscreteFF

import math
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

torch.set_default_device(device)



class ForwardReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState = None):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return previous_action[0]

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return previous_action[0]


path = os.getcwd()
#Kickoff,TouchAvoiding,TouchFarming
folder_path = os.path.join(path, 'TouchFarming')
#76, spawn opponents False, default state, single action loop
#107, spawn opponents True, random state, multi action loop
model = MultiDiscreteFF(107, [512,512,256,256,256], device)

model.load_state_dict(torch.load(os.path.join(folder_path, "PPO_POLICY.pt"),map_location=device))
print(model)
model.eval()


conditions = [TimeoutCondition(300), GoalScoredCondition()]


start_state = RandomState(ball_rand_speed = False, cars_rand_speed = False, cars_on_ground = True)#DefaultState()#


rewardfunc = VelocityPlayerToBallReward()


env = rlgym.make(game_speed=1,
                 raise_on_crash=True, 
                 terminal_conditions = conditions, 
                 state_setter=start_state, 
                 action_parser=DiscreteAction(),
                 reward_fn=rewardfunc,
                 obs_builder=AdvancedObs(),
                 spawn_opponents=True,
                 team_size=1
                 )

episodes = 5

for e in range(1,episodes+1):
    obs = env.reset()
    done = False

    total_reward = 0

    while not done:
        #actions = model.get_action(obs, deterministic=True)[0]
        actions = []
        for ob in obs:
            actions.append(model.get_action(ob, deterministic=True)[0])
        next_obs, rewards, done, _ = env.step(np.array(actions))
        
        obs = next_obs
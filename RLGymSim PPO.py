from rlgym_ppo import Learner
import wandb

import rlgym_sim
from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition,GoalScoredCondition
from rlgym_sim.utils.state_setters import StateSetter, RandomState, DefaultState
from rlgym_sim.utils.action_parsers import DiscreteAction
from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import *

from rlgym_sim.utils.terminal_conditions import TerminalCondition
from rlgym_sim.utils.gamestates import GameState

from rlgym_sim.utils.obs_builders import ObsBuilder,AdvancedObs
from rlgym_sim.utils.gamestates import PlayerData, GameState
from rlgym_tools.extra_rewards.kickoff_reward import KickoffReward
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter
from rlgym_tools.extra_state_setters.goalie_state import GoaliePracticeState
from rlgym_ppo.util import MetricsLogger
from rlgym_sim.utils.common_values import *
import math
import os
import numpy as np

CAR_MASS = 180
GRAVITY = 650

class EnergyReward(RewardFunction):
    # 
    def __init__(self, energy_reward_weight=1):
        super().__init__()
        self.energy_reward_weight = energy_reward_weight

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        max_energy = (CAR_MASS * GRAVITY * (CEILING_Z - 17)) + (0.5 * CAR_MASS * (CAR_MAX_SPEED * CAR_MAX_SPEED))
        energy = 0

        # Add potential energy due to height
        energy += 1.1 * CAR_MASS * GRAVITY * player.car_data.position[2]

        # Add kinetic energy due to velocity
        velocity = np.linalg.norm(player.car_data.linear_velocity)
        energy += 0.5 * CAR_MASS * (velocity * velocity)

        # Add energy contribution from boost
        energy += 7.97e5 * player.boost_amount * 100

        # Add energy contribution from jump
        if player.has_jump:
            energy += 0.8 * 0.5 * CAR_MASS * (292 * 292)

        # Add energy contribution from flip/dodge
        if player.has_flip:
            dodge_impulse = 500 + (velocity / 17) if velocity <= 1700 else (600 - (velocity - 1700))
            dodge_impulse = max(dodge_impulse - 25, 0)
            energy += 0.9 * 0.5 * CAR_MASS * (dodge_impulse * dodge_impulse)

        norm_energy = energy / max_energy
        if player.is_demoed:
            norm_energy = 0

        normalized_reward = norm_energy * self.energy_reward_weight

        return normalized_reward


class ForwardReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState = None):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return previous_action[0]

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return previous_action[0]

class NonConstantReward(RewardFunction):
    def __init__(self, rew_func, steps=5):
        super().__init__()
        self.initial_steps = steps
        self.steps = 0
        self.rew_func = rew_func

    def reset(self, initial_state: GameState = None):
        self.steps = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if self.steps != 0:
            self.steps -= 1
            return 0
        else:
            out = self.rew_func.get_reward(player,state,previous_action)
            self.steps = self.initial_steps
            return out


class Logger(MetricsLogger):
    def __init__(self):
        self.blue_score = 0
        self.orange_score = 0

    def _collect_metrics(self, game_state: GameState) -> list:
        touches = 0
        for player in game_state.players:
            touches += int(player.ball_touched)
        metrics = [np.array([int(self.blue_score < game_state.blue_score), int(self.orange_score < game_state.orange_score)]),
                   touches
        ]
        self.blue_score = game_state.blue_score
        self.orange_score = game_state.orange_score
        return metrics.copy()

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        goals = np.zeros(2)
        touches = 0
        for metric_array in collected_metrics:
            goals += metric_array[0]
            touches += metric_array[1]
        report = {
            "Blue Goals":goals[0],
            "Orange Goals":goals[1],
            "Total Goals":goals[0]+goals[1],
            "Touches":touches,
            "Cumulative Timesteps":cumulative_timesteps}
        wandb_run.log(report)




class EnvFunc:
    def __init__(self):

        self.conditions = [TimeoutCondition(2000), GoalScoredCondition()]

        self.start_state = WeightedSampleSetter(
            (
                DefaultState(),
                RandomState(ball_rand_speed = False, cars_rand_speed = False, cars_on_ground = True)#,
                #GoaliePracticeState(aerial_only=False, allow_enemy_interference=True, first_defender_in_goal=False, reset_to_max_boost=True)
            ),
            (0.1,0.9)#,0.1)
        )

        self.rewardfunc = CombinedReward(
            (
                VelocityPlayerToBallReward(),
                VelocityBallToGoalReward(),
                EventReward(
                    team_goal=5000,
                    concede=-5000,
                    shot=500,
                    save=500,
                ),
                FaceBallReward(),
                NonConstantReward(TouchBallReward()),
                AlignBallGoal(),
                KickoffReward(),
                ForwardReward(),
                SaveBoostReward(),
                EnergyReward()

            ),
            (60, 200, 100, 20, 1000, 20, 100, 1, 1, 20),
        )

        """CombinedReward(
            (
                VelocityPlayerToBallReward(),
                VelocityBallToGoalReward(),
                EventReward(
                    team_goal=10000.0,
                    concede=-10000.0,
                    shot=10.0,
                    save=60.0,
                    demo=20.0,
                ),
                BallYCoordinateReward(),
                FaceBallReward(),
                TouchBallReward(),
                AlignBallGoal(),
                KickoffReward(),
            ),
            (50.0, 10.0, 100.0, 0.1, 0.2, 10.0, 20.0, 10.0),
        )"""

        self.action_parser = DiscreteAction()
        self.obs = AdvancedObs()
    
    def get_notes(self):
        notes = "Rewards:\n"
        for func, weight in zip(self.rewardfunc.reward_functions,self.rewardfunc.reward_weights):
            notes += f"{func.__class__.__name__}: {weight}\n"
            if func.__class__.__name__ == "EventReward":
                for i,c in enumerate(["goal", "team_goal", "concede", "touch", "shot", "save", "demo", "boost_pickup"]):
                    notes += f" - {c}: {func.weights[i]}\n"
        return notes
    
    def __call__(self):
        return rlgym_sim.make(terminal_conditions = self.conditions,
                              state_setter=self.start_state,
                              action_parser=self.action_parser,
                              reward_fn=self.rewardfunc,
                              obs_builder=self.obs,
                              spawn_opponents=True,
                              team_size=1
                              )


if __name__=="__main__":

    build_env = EnvFunc()

    metrics_logger = Logger()


    wandb_run = wandb.init(
        project="rlgym", group="PPO", name="rlgym-ppo-run", reinit=True, notes=build_env.get_notes(), dir="C:/Users/keega/Documents"
    )


    learner = Learner(
        build_env,
        n_proc = 32,
        min_inference_size = 32,
        metrics_logger = metrics_logger,
        ppo_batch_size = 50000,
        ts_per_iteration = 50000,
        exp_buffer_size = 150000,
        policy_layer_sizes = (512, 512, 256, 256, 256),
        critic_layer_sizes = (512, 512, 256, 256, 256),
        ppo_minibatch_size = 50000,
        ppo_ent_coef = 0.001,
        ppo_epochs = 1,
        standardize_returns = True,
        standardize_obs = False,
        save_every_ts = 10_000_000,
        timestep_limit = 1_000_000_000,
        n_checkpoints_to_keep = 50,
        log_to_wandb = True,
        wandb_run = wandb_run,
        )


    learner.learn()
# Fix error which might appear when this script is ran from the command line

import sys
import os

is_running_from_command_line = len(sys.path) <= 7
if is_running_from_command_line:
    script_path_tokens = sys.path[0].split(os.sep)
    sys.path.extend([os.path.join('/', *script_path_tokens[:-1]),
                     os.path.join('/', *script_path_tokens[:-2], 'evoman_framework', 'evoman')])

# Fix evoman resources loading not working because framework assumes the current working directory is
# always `evoman_framework`


if is_running_from_command_line:
    os.chdir('./evoman_framework')
else:
    os.chdir('../../../evoman_framework')

# Start of code without hacks

import torch
import numpy as np
from texttable import Texttable

from test_models.reinforcement_learning.evoman_reinforcement_learning.player_controller import \
    TestReinforcementLearningEvomanPlayerController
from evoman_wrapper.environment_wrapper import EvomanEnvironmentWrapper
import test_models.reinforcement_learning.parameters as parameters

model = torch.load(os.path.join(parameters.MODEL_PATH, 'pyt_save', 'model.pt'))

gains = []
results = []

for enemy in parameters.ENEMIES_CHOSEN_FOR_TESTING:
    min_player_life, max_player_life, average_player_life = 100, 0, 0
    min_enemy_life, max_enemy_life, average_enemy_life = 100, 0, 0
    min_time, max_time, average_time = 100000000, 0, 0

    for experiment in range(parameters.NR_EXPERIMENTS_FOR_EACH_ENEMY):
        evoman_environment = EvomanEnvironmentWrapper('evoman rl test',
                                                      player_controller=TestReinforcementLearningEvomanPlayerController(),
                                                      enemies=[enemy])
        _, player_life, enemy_life, time = evoman_environment.play(pcont=model)

        min_player_life = min(min_player_life, player_life)
        max_player_life = max(max_player_life, player_life)

        min_enemy_life = min(min_enemy_life, enemy_life)
        max_enemy_life = max(max_enemy_life, enemy_life)

        min_time = min(min_time, time)
        max_time = max(max_time, time)

        average_player_life += player_life
        average_enemy_life += enemy_life
        average_time += average_time

    average_player_life /= parameters.NR_EXPERIMENTS_FOR_EACH_ENEMY
    average_enemy_life /= parameters.NR_EXPERIMENTS_FOR_EACH_ENEMY
    average_time /= parameters.NR_EXPERIMENTS_FOR_EACH_ENEMY

    gains.append(100.01 + average_player_life - average_enemy_life)
    results.append(
        [enemy, gains[-1], average_player_life, average_enemy_life, average_time, min_player_life, max_player_life,
         min_enemy_life, max_enemy_life, min_time, max_time])

harmonic_mean_of_gains = len(gains) / np.sum(1.0 / np.array(gains))
print(f'This model has a score for the competition of {harmonic_mean_of_gains:.2f}/200.01\n')

results.append(['Harmonic mean', f'{harmonic_mean_of_gains:.2f}', '-', '-', '-', '-', '-', '-', '-', '-', '-'])

t = Texttable()
t.set_max_width(0)

t.add_rows([['Enemy', 'Gain', 'Avg\nplayer\nlife', 'Avg\nenemy\nlife', 'Avg\nduration',
             'Min\nplayer\nlife', 'Max\nplayer\nlife', 'Min\nenemy\nlife', 'Max\nenemy\nlife', 'Min\nduration',
             'Max\nduration']] + results)
print(t.draw())

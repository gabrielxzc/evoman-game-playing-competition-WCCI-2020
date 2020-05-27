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

for enemy in range(1, 9):
    evoman_environment = EvomanEnvironmentWrapper('evoman rl test',
                                                  player_controller=TestReinforcementLearningEvomanPlayerController(),
                                                  enemies=[enemy],
                                                  level=5)
    _, player_life, enemy_life, time = evoman_environment.play(pcont=model)
    gains.append(100.01 + player_life - enemy_life)
    results.append([enemy, player_life, enemy_life])

print(f'\nThis model has a score for the competition of {len(gains) / np.sum(1.0 / np.array(gains)):.2f}/200.01\n')

t = Texttable()
t.add_rows([['Enemy', 'Player life', 'Enemy life']] + results)
print(t.draw())

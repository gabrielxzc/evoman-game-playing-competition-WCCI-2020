# Fix error which might appear when this script is ran from the command line

import sys
import os

import sys
import os
from pprint import pprint

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
import csv

from test_models.reinforcement_learning.evoman_reinforcement_learning.player_controller import \
    TestReinforcementLearningEvomanPlayerController
from evoman_wrapper.environment_wrapper import EvomanEnvironmentWrapper


def save_csv_results(model_name, results):
    with open(f'../reports/{model_name}.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(
            ['enemy', 'gain', 'avg_player_life', 'avg_enemy_life', 'avg_duration', 'min_player_life', 'max_player_life',
             'min_enemy_life', 'max_enemy_life', 'min_duration', 'max_duration'])
        csvwriter.writerows(results)


def test_model(model_name, ENEMIES_CHOSEN_FOR_TESTING=range(1, 9), NR_EXPERIMENTS_FOR_EACH_ENEMY=30):
    # Start of code without hacks
    MODEL_PATH = f'../trained_models/reinforcement_learning/{model_name}'
    model = torch.load(os.path.join(MODEL_PATH, 'pyt_save', 'model.pt'))

    gains = []
    results = []

    for enemy in ENEMIES_CHOSEN_FOR_TESTING:
        min_player_life, max_player_life, average_player_life = 100, 0, 0
        min_enemy_life, max_enemy_life, average_enemy_life = 100, 0, 0
        min_time, max_time, average_time = 100000000, 0, 0

        for experiment in range(NR_EXPERIMENTS_FOR_EACH_ENEMY):
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
            average_time += time

        average_player_life /= NR_EXPERIMENTS_FOR_EACH_ENEMY
        average_enemy_life /= NR_EXPERIMENTS_FOR_EACH_ENEMY
        average_time /= NR_EXPERIMENTS_FOR_EACH_ENEMY

        gains.append(100.01 + average_player_life - average_enemy_life)
        results.append(
            [enemy, gains[-1], average_player_life, average_enemy_life, average_time, min_player_life, max_player_life,
             min_enemy_life, max_enemy_life, min_time, max_time])

    save_csv_results(model_name, results)  # always `evoman_framework`

    harmonic_mean_of_gains = len(gains) / np.sum(1.0 / np.array(gains))
    return harmonic_mean_of_gains


def test_models(models):
    results = dict()
    for model in models:
        try:
            result = test_model(model)
            results[model] = result
        except Exception as e:
            print(model, e)
    pprint(results)


if __name__ == '__main__':
    test_models(['1593196354.556512',
                 '1593241593.7984622',
                 '1593247383.7798543',
                 '1593253419.8104591',
                 '1593259634.755718',
                 '1593265879.0949752',
                 '1593272302.7187254',
                 '1593278800.13606',
                 '1593285197.2534902',
                 '1593291408.720644',
                 '1593297821.159123',
                 '1593304049.8160946'])

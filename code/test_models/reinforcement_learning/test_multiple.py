# Fix error which might appear when this script is ran from the command line
import json
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
             'min_enemy_life', 'max_enemy_life', 'min_duration', 'max_duration', 'percentage_games_lost'])
        csvwriter.writerows(results)


def test_model(model_name, ENEMIES_CHOSEN_FOR_TESTING=range(1, 9), NR_EXPERIMENTS_FOR_EACH_ENEMY=30):
    # Start of code without hacks
    MODEL_PATH = f'../trained_models/reinforcement_learning/{model_name}'
    model = torch.load(os.path.join(MODEL_PATH, 'pyt_save', 'model.pt'))

    with open(os.path.join(MODEL_PATH, 'config.json')) as data_file:
        data = json.load(data_file)
        # try:
        #     difficulty = data['enemies_difficulty']
        # except Exception as e:
        #     difficulty = 2
        ENEMIES_CHOSEN_FOR_TESTING = data['enemies']
    difficulty = 5
    print(model_name, ENEMIES_CHOSEN_FOR_TESTING)
    gains = []
    results = []

    for enemy in ENEMIES_CHOSEN_FOR_TESTING:
        min_player_life, max_player_life, average_player_life = 100, 0, 0
        min_enemy_life, max_enemy_life, average_enemy_life = 100, 0, 0
        min_time, max_time, average_time = 100000000, 0, 0
        number_of_games_lost = 0

        for experiment in range(NR_EXPERIMENTS_FOR_EACH_ENEMY):
            evoman_environment = EvomanEnvironmentWrapper('evoman rl test',
                                                          player_controller=TestReinforcementLearningEvomanPlayerController(),
                                                          enemies=[enemy],
                                                          level=difficulty)
            _, player_life, enemy_life, time = evoman_environment.play(pcont=model)

            number_of_games_lost += 1 if player_life == 0 else 0

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
        percentage_of_games_lost = 100 * number_of_games_lost / NR_EXPERIMENTS_FOR_EACH_ENEMY

        gains.append(100.01 + average_player_life - average_enemy_life)
        results.append(
            [enemy, gains[-1], average_player_life, average_enemy_life, average_time, min_player_life, max_player_life,
             min_enemy_life, max_enemy_life, min_time, max_time, percentage_of_games_lost])

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
    test_models(list(sorted(os.listdir('../trained_models/reinforcement_learning'))[-1:]))

# Fix error which might appear when this script is ran from the command line

import sys
import os

is_running_from_command_line = len(sys.path) <= 6
if is_running_from_command_line:
    script_path_tokens = sys.path[0].split(os.sep)
    sys.path.extend([os.path.join('/', *script_path_tokens[:-1]),
                     os.path.join('/', *script_path_tokens[:-2], 'evoman_framework', 'evoman')])

# Fix evoman resources loading not working because framework assumes the current working directory is
# always `evoman_framework`


if is_running_from_command_line:
    os.chdir('./evoman_framework')
else:
    os.chdir('../../evoman_framework')

# Start of code without hacks


import pso_bootstrap.parameters as parameters

from pso_bootstrap.evoman_pso.algorithm import EvomanPsoAlgorithm
from pso_bootstrap.evoman_pso.parameters import EvomanPsoParameters

from utils.pickle import save_class_instance

if __name__ == '__main__':
    evoman_pso_parameters = EvomanPsoParameters(
        enemies_chosen_for_training=parameters.ENEMIES_CHOSEN_FOR_TRAINING,
        enemies_difficulty_level=parameters.ENEMIES_DIFFICULTY_LEVEL,
        model_hidden_layers_sizes=parameters.MODEL_HIDDEN_LAYERS_SIZES,
        model_min_weight=parameters.MODEL_MIN_WEIGHT,
        model_max_weight=parameters.MODEL_MAX_WEIGHT,
        pop_size=parameters.POP_SIZE,
        nr_iterations=parameters.NR_ITERATIONS,
        inertia_initial_weight=parameters.INERTIA_INITIAL_WEIGHT,
        cognitive_weight=parameters.COGNITIVE_WEIGHT,
        social_weight=parameters.SOCIAL_WEIGHT,
        min_particle_speed=parameters.MIN_PARTICLE_SPEED,
        max_particle_speed=parameters.MAX_PARTICLE_SPEED,
        thread_pool_size=parameters.THREAD_POOL_SIZE
    )

    evoman_pso_algorithm = EvomanPsoAlgorithm(evoman_pso_parameters)
    evoman_pso_solution = evoman_pso_algorithm.train()

    evoman_pso_parameters.enemies_chosen_for_training.sort()
    pso_bootstrap_trained_models_dir = f'../trained_models/pso_bootstrap/{evoman_pso_parameters.enemies_chosen_for_training}_{evoman_pso_parameters.enemies_difficulty_level}'
    pso_bootstrap_trained_model_name = f'{evoman_pso_solution.model_fitness:.2f}'

    save_class_instance(evoman_pso_solution, pso_bootstrap_trained_models_dir, pso_bootstrap_trained_model_name,
                        is_with_timestamp=True)

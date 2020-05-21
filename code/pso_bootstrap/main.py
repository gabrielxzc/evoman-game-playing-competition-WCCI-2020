# Fix for evoman images import not working because framework assumes the current working directory
# is always `evoman_framework`
import os

os.chdir('../../evoman_framework')

# Start of code without hacks

import pso_bootstrap.parameters as parameters

from pso_bootstrap.evoman_pso.algorithm import EvomanPsoAlgorithm
from pso_bootstrap.evoman_pso.parameters import EvomanPsoParameters

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
    evoman_pso_algorithm.train()

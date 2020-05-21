import numpy as np

import evoman_wrapper.constants as evoman_constants
from evoman_wrapper.environment_wrapper import EvomanEnvironmentWrapper
from evoman_wrapper.keras_model_player_controller import KerasModelPlayerController

from pso_bootstrap.pso.algorithm import PsoAlgorithm
from pso_bootstrap.pso.parameters import PsoParameters

from pso_bootstrap.evoman_pso.parameters import EvomanPsoParameters

import utils.keras as keras_utils


class EvomanPsoAlgorithm:
    def __init__(self, evoman_pso_parameters: EvomanPsoParameters):
        self._evoman_pso_parameters = evoman_pso_parameters

    def train(self):
        pso_parameters = PsoParameters(
            pop_size=self._evoman_pso_parameters.pop_size,
            nr_iterations=self._evoman_pso_parameters.nr_iterations,
            generate_particle_state=self._generate_particle_state,
            generate_particle_speed=self._generate_particle_speed,
            evaluate_particle_state=self._evaluate_particle_state,
            inertia_initial_weight=self._evoman_pso_parameters.inertia_initial_weight,
            cognitive_weight=self._evoman_pso_parameters.cognitive_weight,
            social_weight=self._evoman_pso_parameters.social_weight,
            min_particle_speed=self._evoman_pso_parameters.min_particle_speed,
            max_particle_speed=self._evoman_pso_parameters.max_particle_speed,
            min_particle_state=self._evoman_pso_parameters.model_min_weight,
            max_particle_state=self._evoman_pso_parameters.model_max_weight,
            thread_pool_size=self._evoman_pso_parameters.thread_pool_size
        )

        pso_algorithm = PsoAlgorithm(pso_parameters)
        pso_solution = pso_algorithm.train()

    def _generate_particle_state(self):
        model = self._get_model()
        return keras_utils.get_model_flattened_weights(model)

    def _get_model(self, weights=None):
        return keras_utils.get_model(evoman_constants.OBSERVATION_SPACE_SIZE,
                                     self._get_model_hidden_and_output_layers_sizes(),
                                     weights)

    def _get_model_hidden_and_output_layers_sizes(self):
        return self._evoman_pso_parameters.model_hidden_layers_sizes + [evoman_constants.ACTION_SPACE_SIZE]

    def _generate_particle_speed(self, particle_state):
        return (self._evoman_pso_parameters.max_particle_speed - self._evoman_pso_parameters.min_particle_speed) * \
               np.random.sample(particle_state.shape) + \
               self._evoman_pso_parameters.min_particle_speed

    def _evaluate_particle_state(self, particle_state):
        model_flattened_weights = particle_state
        weights = keras_utils.get_model_weights_from_flattened_weights(model_flattened_weights,
                                                                       [evoman_constants.OBSERVATION_SPACE_SIZE,
                                                                        *self._get_model_hidden_and_output_layers_sizes()])
        model = self._get_model(weights)

        evoman_environment = EvomanEnvironmentWrapper('evoman pso bootstrap',
                                                      player_controller=KerasModelPlayerController(),
                                                      enemies=self._evoman_pso_parameters.enemies_chosen_for_training,
                                                      multiplemode="yes",
                                                      level=self._evoman_pso_parameters.enemies_difficulty_level)

        fitness, _, _, _ = evoman_environment.play(pcont=model)
        return fitness

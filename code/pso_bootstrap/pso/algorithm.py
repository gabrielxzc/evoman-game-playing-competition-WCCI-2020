import multiprocessing
import math

import numpy as np

from pso_bootstrap.pso.parameters import PsoParameters
from pso_bootstrap.pso.particle import PsoParticle
from pso_bootstrap.pso.solution import PsoSolution


class PsoAlgorithm:
    def __init__(self, pso_parameters: PsoParameters):
        self._pso_parameters = pso_parameters

    def train(self):
        thread_pool = self._setup_train()

        self._generate_population()

        for self._iteration_idx in range(self._pso_parameters.nr_iterations):
            self._evaluate_population(thread_pool)
            self._update_all_time_and_individual_best_particles()

            self._print_iteration_details()

            for particle in self._population:
                self._update_particle_speed(particle)
                self._update_particle_state(particle)

            self._adjust_inertia_weight()

        self._tear_down_train(thread_pool)

        return PsoSolution(self._all_time_best_particle_state, self._all_time_best_particle_fitness,
                           self._pso_parameters)

    def _setup_train(self):
        self._inertia_weight = self._pso_parameters.inertia_initial_weight

        self._all_time_best_particle_state = None
        self._all_time_best_particle_fitness = -math.inf

        thread_pool = multiprocessing.Pool(processes=self._pso_parameters.thread_pool_size)

        return thread_pool

    def _generate_population(self):
        self._population = []

        for _ in range(self._pso_parameters.pop_size):
            particle_state = self._pso_parameters.generate_particle_state()
            particle_speed = self._pso_parameters.generate_particle_speed(particle_state)

            self._population.append(PsoParticle(particle_state, particle_speed))

    def _evaluate_population(self, pool):
        fitnesses = pool.map(self._evaluate_particle, self._population)

        for particle_idx, fitness in enumerate(fitnesses):
            self._population[particle_idx].fitness = fitness

    def _evaluate_particle(self, particle):
        return self._pso_parameters.evaluate_particle_state(particle.state)

    def _update_all_time_and_individual_best_particles(self):
        self._current_iteration_best_particle_state = None
        self._current_iteration_best_particle_fitness = -math.inf

        for particle in self._population:
            self._update_all_time_best_particle(particle)
            self._update_current_iteration_best_particle(particle)
            self._update_individual_particle_best(particle)

    def _update_all_time_best_particle(self, particle):
        if self._all_time_best_particle_fitness < particle.fitness:
            self._all_time_best_particle_state = particle.state
            self._all_time_best_particle_fitness = particle.fitness

    def _update_current_iteration_best_particle(self, particle):
        if self._current_iteration_best_particle_fitness < particle.fitness:
            self._current_iteration_best_particle_state = particle.state
            self._current_iteration_best_particle_fitness = particle.fitness

    def _update_individual_particle_best(self, particle):
        if particle.best_fitness < particle.fitness:
            particle.best_state = particle.state
            particle.best_fitness = particle.fitness

    def _print_iteration_details(self):
        print(f'iteration: {self._iteration_idx} \t best_fitness: {self._all_time_best_particle_fitness}')

    def _update_particle_speed(self, particle):
        cognitive_random_weights = self._random_particle_speed_weights(particle)
        cognitive_weight = self._pso_parameters.cognitive_weight

        social_random_weights = self._random_particle_speed_weights(particle)
        social_weight = self._pso_parameters.social_weight

        current_iteration_best_state = self._current_iteration_best_particle_state

        unclipped_speed = self._inertia_weight * particle.speed + \
                          cognitive_weight * cognitive_random_weights * (particle.best_state - particle.state) + \
                          social_weight * social_random_weights * (current_iteration_best_state - particle.state)

        particle.speed = np.clip(unclipped_speed, self._pso_parameters.min_particle_speed,
                                 self._pso_parameters.max_particle_speed)

    def _random_particle_speed_weights(self, particle):
        return 2 * np.random.random_sample(particle.speed.shape)

    def _update_particle_state(self, particle):
        unclipped_state = particle.state + particle.speed
        particle.state = np.clip(unclipped_state, self._pso_parameters.min_particle_state,
                                 self._pso_parameters.max_particle_state)

    def _adjust_inertia_weight(self):
        self._inertia_weight = (1 - (self._iteration_idx / self._pso_parameters.nr_iterations)) * 0.7 + 0.3

    def _tear_down_train(self, thread_pool):
        thread_pool.close()

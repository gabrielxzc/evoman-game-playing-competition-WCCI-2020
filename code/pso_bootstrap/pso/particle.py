import math


class PsoParticle:
    def __init__(self, initial_state, initial_speed):
        self.state = initial_state
        self.fitness = -math.inf

        self.best_state = initial_state
        self.best_fitness = self.fitness

        self.speed = initial_speed

class EvomanPsoSolution:
    def __init__(self, model_flattened_weights, model_fitness, evoman_pso_parameters, time_to_train_in_seconds):
        self.model_flattened_weights = model_flattened_weights
        self.model_fitness = model_fitness
        self.evoman_pso_parameters = evoman_pso_parameters
        self.time_to_train_in_seconds = time_to_train_in_seconds

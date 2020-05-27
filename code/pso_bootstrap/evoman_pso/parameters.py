class EvomanPsoParameters:
    def __init__(self, enemies_chosen_for_training, model_hidden_layers_sizes, model_min_weight, model_max_weight,
                 pop_size, nr_iterations, inertia_initial_weight, cognitive_weight, social_weight, min_particle_speed,
                 max_particle_speed, thread_pool_size=1):
        assert isinstance(enemies_chosen_for_training, list) and len(enemies_chosen_for_training) == 4, \
            'As stated in the competition rules, 4 enemies must be chosen for training'

        self.enemies_chosen_for_training = enemies_chosen_for_training

        self.model_hidden_layers_sizes = model_hidden_layers_sizes
        self.model_min_weight = model_min_weight
        self.model_max_weight = model_max_weight

        self.pop_size = pop_size
        self.nr_iterations = nr_iterations

        self.inertia_initial_weight = inertia_initial_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

        self.min_particle_speed = min_particle_speed
        self.max_particle_speed = max_particle_speed

        self.thread_pool_size = thread_pool_size

class PsoParameters:
    def __init__(self, pop_size, nr_iterations, generate_particle_state, generate_particle_speed,
                 evaluate_particle_state, inertia_initial_weight, cognitive_weight, social_weight,
                 min_particle_speed, max_particle_speed, min_particle_state, max_particle_state, thread_pool_size=1):
        self.pop_size = pop_size
        self.nr_iterations = nr_iterations

        self.generate_particle_state = generate_particle_state
        self.generate_particle_speed = generate_particle_speed
        self.evaluate_particle_state = evaluate_particle_state

        self.inertia_initial_weight = inertia_initial_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

        self.min_particle_speed = min_particle_speed
        self.max_particle_speed = max_particle_speed

        self.min_particle_state = min_particle_state
        self.max_particle_state = max_particle_state

        self.thread_pool_size = thread_pool_size

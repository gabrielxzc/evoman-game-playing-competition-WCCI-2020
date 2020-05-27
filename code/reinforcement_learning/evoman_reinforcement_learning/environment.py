from evoman_wrapper.environment_wrapper import EvomanEnvironmentWrapper


class ReinforcementLearningEvomanEnvironment(EvomanEnvironmentWrapper):
    def __init__(self, experiment_name='test', **kwargs):
        super().__init__(experiment_name, **kwargs)

        self.player_previous_life = None
        self.enemy_previous_life = None

    def play(self, pcont="None", econt="None"):
        self.player_previous_life = 100
        self.enemy_previous_life = 100

        return super().play(pcont, econt)

    def get_reward(self):
        reward = (self.player.life - self.player_previous_life) + (self.enemy_previous_life - self.enemy.life)

        self.player_previous_life = self.player.life
        self.enemy_previous_life = self.enemy.life

        return reward

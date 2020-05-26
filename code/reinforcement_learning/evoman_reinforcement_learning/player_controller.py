import torch
import numpy as np

from controller import Controller as EvomanPlayerController


class ReinforcementLearningEvomanPlayerController(EvomanPlayerController):
    def __init__(self, evoman_environment, buf, logger):
        self._evoman_environment = evoman_environment
        self._buf = buf
        self._logger = logger

        self.are_all_timesteps_saved = True
        self.first_not_saved_observation = None

        self._is_first_observation = True
        self.previous_observation = None
        self._previous_action = None
        self._previous_v = None
        self._previous_logp = None

        self.episode_ret = 0
        self.episode_len = 0

    def control(self, observation, ac=None):
        a, v, logp = ac.step(torch.as_tensor(observation, dtype=torch.float32))
        self._logger.store(VVals=v)
        self.episode_len += 1

        if self._is_first_observation:
            self._is_first_observation = False
        else:
            self._update_buf()

        self.previous_observation = np.array(observation, dtype=np.float32)
        self._previous_action = a
        self._previous_v = v
        self._previous_logp = logp

        return [int(bit) for bit in bin(a)[2:].zfill(5)]

    def _update_buf(self):
        try:
            reward = self._evoman_environment.get_reward()
            self._buf.store(self.previous_observation, self._previous_action,
                            reward, self._previous_v, self._previous_logp)

            self.episode_ret += reward
        except AssertionError:
            if self.are_all_timesteps_saved:
                self.first_not_saved_observation = self.previous_observation

            self.are_all_timesteps_saved = False

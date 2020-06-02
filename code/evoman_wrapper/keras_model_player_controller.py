import torch
import torch.distributions as distributions
import numpy as np

from controller import Controller


class KerasModelPlayerController(Controller):
    def control(self, observation, model=None):
        action = self._get_action(observation, model)
        return action

    def _get_action(self, observation, model):
        prediction = model.predict(np.array([observation]))[0]
        return self._get_action_from_prediction(prediction)

    def _get_action_from_prediction(self, prediction):
        action = distributions.Categorical(logits=torch.as_tensor(prediction, dtype=torch.float32)).sample()
        return [int(bit) for bit in bin(action)[2:].zfill(5)]

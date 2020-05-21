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
        get_action_from_prediction_vectorize = np.vectorize(lambda x: 0 if x < 0 else 1)
        return get_action_from_prediction_vectorize(prediction)

import torch

from controller import Controller as EvomanPlayerController


class TestReinforcementLearningEvomanPlayerController(EvomanPlayerController):
    def control(self, observation, ac=None):
        a, _, _ = ac.step(torch.as_tensor(observation, dtype=torch.float32))
        return [int(bit) for bit in bin(a)[2:].zfill(5)]

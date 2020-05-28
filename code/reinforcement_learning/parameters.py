import torch

ENEMIES_CHOSEN_FOR_TRAINING = [1, 2, 6, 7]

MODEL_HIDDEN_LAYERS_SIZES = (64, 64)
MODEL_ACTIVATION = torch.nn.Tanh

STEPS_PER_EPOCH = 10000
EPOCHS = 2000

STARTING_MODEL_PATH = '../trained_models/reinforcement_learning/1590579471.1575735'

NR_PARALLEL_PROCESSES = 3

GAMMA = 0.99
CLIP_RATIO = 0.2
PI_LR = 3e-4
VF_LR = 1e-3
TRAIN_PI_ITERATIONS = 80
TRAIN_V_ITERATIONS = 80
LAMBDA = 0.97
TARGET_KL = 0.01
SAVE_FREQ = 10

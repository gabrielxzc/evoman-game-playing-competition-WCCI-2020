import torch
import os

# [1, 2, 6, 7] 127.54
# [3, 4, 5, 8]
# [3, 4, 6, 7]
# [1, 3, 6, 7] 100
# [2, 3, 6, 7] 137 ! incorrect
# [2, 4, 6, 7] 87.13

# corrected
# [2, 3, 6, 7] 95.69

ENEMIES_CHOSEN_FOR_TRAINING = [3]
ENEMIES_DIFFICULTY = 2

MODEL_HIDDEN_LAYERS_SIZES = (64, 64)
MODEL_ACTIVATION = torch.nn.Tanh

STEPS_PER_EPOCH = 10000
EPOCHS = 1000  # 2000

# trained_models/pso_bootstrap/[1, 2, 6, 7]/47.98_1591201550.222573.pickle 43.2
# trained_models/pso_bootstrap/[1, 2, 6, 7]/52.13_1591215640.939492.pickle 42.7
# trained_models/pso_bootstrap/[1, 2, 6, 7]/56.86_1591208496.467009.pickle 54.36
# trained_models/pso_bootstrap/[1, 2, 6, 7]/61.38_1591194194.380356.pickle 89.16

# STARTING_MODEL_PATH = '../trained_models/pso_bootstrap/[1, 2, 6, 7]/61.38_1591194194.380356.pickle'
# STARTING_MODEL_PATH = '../trained_models/reinforcement_learning/1591261004.787129'

# model_name = sorted(os.listdir('trained_models/reinforcement_learning'))[-1]
# STARTING_MODEL_PATH = f'../trained_models/reinforcement_learning/{model_name}'

STARTING_MODEL_PATH = None
IS_STARTING_MODEL_PSO = False

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

# Fix evoman resources loading not working because framework assumes the current working directory is
# always `evoman_framework`

import os

if not os.getcwd().endswith('evoman_framework'):
    os.chdir('../../evoman_framework')

# Start of code without hacks

import reinforcement_learning.parameters as parameters

from spinup import ppo_pytorch as ppo
from spinup.utils.mpi_tools import mpi_fork

import time

mpi_fork(parameters.NR_PARALLEL_PROCESSES)

logger_kwargs = dict(output_dir=f'../trained_models/reinforcement_learning/{time.time()}',
                     exp_name='evoman reinforcement learning')
seed = int(time.time())

ppo(ac_kwargs={'hidden_sizes': parameters.MODEL_HIDDEN_LAYERS_SIZES, 'activation': parameters.MODEL_ACTIVATION},
    seed=seed, steps_per_epoch=parameters.STEPS_PER_EPOCH, epochs=parameters.EPOCHS,
    enemies=parameters.ENEMIES_CHOSEN_FOR_TRAINING, enemy_difficulty=parameters.ENEMIES_DIFFICULTY_LEVEL,
    gamma=parameters.GAMMA, clip_ratio=parameters.CLIP_RATIO, pi_lr=parameters.PI_LR, vf_lr=parameters.VF_LR,
    train_pi_iters=parameters.TRAIN_PI_ITERATIONS, train_v_iters=parameters.TRAIN_V_ITERATIONS, lam=parameters.LAMBDA,
    target_kl=parameters.TARGET_KL, logger_kwargs=logger_kwargs, save_freq=parameters.SAVE_FREQ)

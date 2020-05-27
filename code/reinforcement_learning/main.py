# Fix error which might appear when this script is ran from the command line

import sys
import os

is_running_from_command_line = len(sys.path) <= 7
if is_running_from_command_line:
    script_path_tokens = sys.path[0].split(os.sep)
    sys.path.extend([os.path.join('/', *script_path_tokens[:-1]),
                     os.path.join('/', *script_path_tokens[:-2], 'evoman_framework', 'evoman')])

# Parallelization part, had to be done before all the hacks

from spinup.utils.mpi_tools import mpi_fork
import reinforcement_learning.parameters as parameters

mpi_fork(parameters.NR_PARALLEL_PROCESSES)

# Fix evoman resources loading not working because framework assumes the current working directory is
# always `evoman_framework`

if is_running_from_command_line:
    if not os.getcwd().endswith('evoman_framework'):
        os.chdir('./evoman_framework')
else:
    if not os.getcwd().endswith('evoman_framework'):
        os.chdir('../../evoman_framework')

# Start of code without hacks


from spinup.algos.pytorch.ppo.ppo import ppo

import time

logger_kwargs = dict(output_dir=f'../trained_models/reinforcement_learning/{time.time()}',
                     exp_name='evoman reinforcement learning')
seed = int(time.time())

ppo(ac_kwargs={'hidden_sizes': parameters.MODEL_HIDDEN_LAYERS_SIZES, 'activation': parameters.MODEL_ACTIVATION},
    seed=seed, steps_per_epoch=parameters.STEPS_PER_EPOCH, epochs=parameters.EPOCHS,
    enemies=parameters.ENEMIES_CHOSEN_FOR_TRAINING, enemy_difficulty=parameters.ENEMIES_DIFFICULTY_LEVEL,
    gamma=parameters.GAMMA, clip_ratio=parameters.CLIP_RATIO, pi_lr=parameters.PI_LR, vf_lr=parameters.VF_LR,
    train_pi_iters=parameters.TRAIN_PI_ITERATIONS, train_v_iters=parameters.TRAIN_V_ITERATIONS, lam=parameters.LAMBDA,
    target_kl=parameters.TARGET_KL, logger_kwargs=logger_kwargs, save_freq=parameters.SAVE_FREQ,
    starting_actor_critic=parameters.STARTING_MODEL_PATH)

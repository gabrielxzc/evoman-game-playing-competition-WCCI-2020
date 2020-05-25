# Fix evoman resources loading not working because framework assumes the current working directory is
# always `evoman_framework`

import os

if not os.getcwd().endswith('evoman_framework'):
    os.chdir('../../evoman_framework')

# Start of code without hacks

from spinup.utils.mpi_tools import mpi_fork
from spinup import ppo_pytorch as ppo

mpi_fork(6)

logger_kwargs = dict(output_dir='path/to/output_dir', exp_name='experiment_name')
ppo(steps_per_epoch=10000, epochs=500, logger_kwargs=logger_kwargs)

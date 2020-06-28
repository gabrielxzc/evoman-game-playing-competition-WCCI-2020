import os

ENEMIES_CHOSEN_FOR_TESTING = range(1, 9)
ENEMIES_DIFFICULTY = 2
NR_EXPERIMENTS_FOR_EACH_ENEMY = 30

# model_name = sorted(os.listdir('../trained_models/reinforcement_learning'))[-1]
# model_name = '1591964820.216715'  # 124.23
# model_name = '1591641901.9027252'  # ppo run 1 3000
# model_name = '1591880695.9546342'  # ppo run 2 3000
# model_name = '1591972086.6519217'  # ppo run 3 3000
model_name = '1591935751.1014764'  # ppo run 3 1750
MODEL_PATH = f'../trained_models/reinforcement_learning/{model_name}'

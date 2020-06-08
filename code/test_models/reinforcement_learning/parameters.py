import os

ENEMIES_CHOSEN_FOR_TESTING = range(1, 9)
NR_EXPERIMENTS_FOR_EACH_ENEMY = 30

model_name = sorted(os.listdir('../trained_models/reinforcement_learning'))[-1]
# model_name = '1591261004.787129'  # 124.23
MODEL_PATH = f'../trained_models/reinforcement_learning/{model_name}'

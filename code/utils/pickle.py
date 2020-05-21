import os
import pickle
from pathlib import Path

from utils.time import now_timestamp


def save_class_instance(class_instance, dir_path, file_name, is_with_timestamp=True):
    Path(dir_path).mkdir(parents=True, exist_ok=True)

    file_name = f'{file_name}{"_" + str(now_timestamp()) if is_with_timestamp else ""}'
    file_path = os.path.join(dir_path, file_name + '.pickle')

    with open(file_path, 'wb') as file:
        pickle.dump(class_instance, file)

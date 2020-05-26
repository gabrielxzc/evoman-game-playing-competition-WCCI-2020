from environment import Environment as EvomanEnvironment


class EvomanEnvironmentWrapper(EvomanEnvironment):
    def __init__(self, experiment_name, **kwargs):
        is_speed_normal = "speed" in kwargs and kwargs["speed"] == "normal"
        logs = "on" if is_speed_normal else "off"

        super().__init__(experiment_name, logs=logs, savelogs="no", **kwargs)

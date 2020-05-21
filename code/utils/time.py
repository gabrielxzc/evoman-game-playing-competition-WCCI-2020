from datetime import datetime


def now_timestamp():
    now = datetime.now()
    return datetime.timestamp(now)

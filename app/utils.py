from os import environ
import time
import logging
from app.constant import Profile

logger = logging.getLogger(__name__)


# Replace multiple string pairs in a string
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)

    return text


def print_red(text):
    print(f"\033[31m{text}\033[0m")


def get_profile():
    try:
        return Profile(environ.get("PROFILE", "local"))
    except ValueError as e:
        raise ValueError(f"profile: {environ.get('PROFILE')} is not a valid profile value") from e


def is_production():
    return get_profile() == Profile.PRODUCTION


def log_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Execution time of function {func.__name__}: {execution_time}s")
        return result

    return wrapper

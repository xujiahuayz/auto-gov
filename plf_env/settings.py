import inspect
import os
from os import path

PROJECT_ROOT = path.dirname(path.dirname(__file__))
LOG_FORMAT = "%(asctime)-15s - %(levelname)s - %(message)s"

CACHE_PATH = path.join(PROJECT_ROOT, ".cache")
DISK_CACHING = True


def _is_test():
    stack = inspect.stack()
    return any(x[0].f_globals["__name__"].startswith("_pytest.") for x in stack)


def _get_env():
    if "PLF_ENV" in os.environ:
        return os.environ["PLF_ENV"]
    if _is_test():
        return "test"
    return "development"


PLF_ENV = _get_env()

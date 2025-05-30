"""
credit: https://github.com/danhper/ethereum-tools/blob/master/eth_tools/caching.py
"""

import functools
import hashlib
import os
import pickle
import time
from pathlib import Path
from typing import Callable, Optional

from market_env.constants import CACHE_PATH


def cache(
    ttl: int,
    min_memory_time: float = 0.1,
    min_disk_time: float = 2.0,
    directory: Path = CACHE_PATH,
    exclude: Optional[dict] = None,
    should_cache=None,
) -> Callable:
    """Decorator using on-disk caching. If the function call takes more than
    ``min_memory_time`` and less than ``min_disk_time``, the result will be
    cached in memory. If the call takes more than ``min_disk_time``, the result
    will be cached inside ``directory`` as a pickle file.
    Regardless of the storage, cached results are used only for ``ttl`` seconds.
    If ``ttl`` is set to a negative value, the cache will never be expired.
    The function result is cached based on its name and the arguments it has
    been passed. This means that if the function is not passed the exact
    same arguments, the result will not be re-used.

    Args:
        ttl (int): Time to live in seconds
        min_memory_time (float): Minimum time in seconds to cache in memory
        min_disk_time (float): Minimum time in seconds to cache on disk
        directory (str): Directory where to store the cache files
        exclude (dict): Dictionary with two keys, ``args`` and ``kwargs``.
            Each key should contain a list of integers or strings. The integers
            represent the indexes of the arguments to be excluded from the
            cache key. The strings represent the names of the keyword arguments
            to be excluded from the cache key.
        should_cache (callable): Function that receives the function result
            and returns a boolean indicating if the result should be cached or
            not. If not provided, the result will always be cached.

    Returns:
        callable: Decorated function
    """
    os.makedirs(CACHE_PATH, 0o755, exist_ok=True)

    if exclude is None:
        exclude = {}

    def compute_key(func_name, args, kwargs):
        """Computes a key to be used for the function result
        depending on the function name and its arguments
        """
        reconstructed_args = []
        for i, arg in enumerate(args):
            if i not in exclude.get("args", []):
                reconstructed_args.append(arg)
        for arg in exclude.get("kwargs", []):
            del kwargs[arg]
        to_hash = pickle.dumps((func_name, reconstructed_args, kwargs))
        md5sum = hashlib.md5()
        md5sum.update(to_hash)
        return md5sum.hexdigest()

    def should_use_file_cache(filepath):
        """Returns ``True`` if the file cache should be used"""
        try:
            stats = os.stat(filepath)
            if 0 <= ttl <= int(time.time() - stats.st_mtime):
                os.unlink(filepath)
                return False
            return True
        except FileNotFoundError:
            return False

    def decorator(fn: Callable) -> Callable:
        memory_cache = {}

        def get_from_memory_cache(key):
            """Returns two values, the second indicates if the value was cached
            or not. If the second value is true, the first value is the
            actual cached value
            """
            entry = memory_cache.get(key)
            if not entry:
                return None, False
            inserted_at, value = entry
            if 0 <= ttl <= time.time() - inserted_at:
                del memory_cache[key]
                return None, False
            return value, True

        def decorated(*args, **kwargs):
            # compute unique key depending on function name and arguments
            key = compute_key(fn.__name__, args, kwargs)

            # return from memory if possible
            value, cached = get_from_memory_cache(key)
            if value is not None and cached:
                return value

            # return from disk if possible
            filename = "{0}.pkl".format(key)
            filepath = Path(directory) / filename
            if should_use_file_cache(filepath):
                with open(filepath, "rb") as f:
                    return pickle.load(f)

            # not cached, run the computation
            start = time.time()
            result = fn(*args, **kwargs)
            ellapsed = time.time() - start

            if should_cache is not None and not should_cache(result, *args, **kwargs):
                return result

            # if ellapsed time is long enough to store in memory
            # short enough not to fallback to disk storage or disk storage is not enabled
            # add to memory cache
            if min_memory_time <= ellapsed < min_disk_time:
                memory_cache[key] = (time.time(), result)

            # if ellapsed time is long, use disk storage instead of memory
            elif ellapsed >= min_disk_time:
                with open(filepath, "wb") as f:
                    pickle.dump(result, f)
            return result

        return functools.update_wrapper(decorated, fn)

    return decorator

import gzip
import heapq
import json
from typing import Callable, Iterator, List, TextIO, TypeVar, Union

from plf_env.constants import SECONDS_PER_YEAR
from plf_env.core import Event

T = TypeVar("T")


def compute_compound_interests(
    rate: float, current_timestamp: int, previous_timestamp: int
):
    # https://github.com/aave/protocol-v2/blob/dbd77ad9312f607b420da746c2cb7385d734b015/contracts/protocol/libraries/math/MathUtils.sol#L45
    exp = current_timestamp - previous_timestamp
    rate_per_seconds = rate / SECONDS_PER_YEAR
    return (1 + rate_per_seconds) ** exp


def compute_linear_interests(
    rate: float, current_timestamp: int, previous_timestamp: int
):
    # https://github.com/aave/protocol-v2/blob/dbd77ad9312f607b420da746c2cb7385d734b015/contracts/protocol/libraries/math/MathUtils.sol#L21
    multiplier = current_timestamp - previous_timestamp
    rate_per_seconds = rate / SECONDS_PER_YEAR
    return 1 + rate_per_seconds * multiplier


def scale(value: Union[int, float], decimals: int) -> float:
    return value / 10 ** decimals


def round_down_hour(timestamp: int) -> int:
    return (timestamp // 3600) * 3600


def find(predicate: Callable[[T], bool], values: List[T]) -> T:
    return [v for v in values if predicate(v)][0]


def stream_events(iterators: List[Iterator[Event]]):
    """`iterators` is a list of iterators (typically each one coming from a single file)
    where each value of an iterator should be an event
    This function assumes that each iterator is sorted in ascending order
    with respect to Event.key
    """
    events = []
    # insert an event from each iterator in a heap
    # the top value of the heap will always be the earliest available event
    for it in iterators:
        event = next(it)
        heapq.heappush(events, (event, it))

    while events:
        event, it = heapq.heappop(events)
        yield event
        try:
            event = next(it)
            heapq.heappush(events, (event, it))
        except StopIteration:
            continue


class EventFilesWrapper:
    def __init__(self, filepaths: List[str]):
        self.filepaths = filepaths
        self.files: List[TextIO] = []

    def __enter__(self):
        iterables = []
        for filepath in self.filepaths:
            fobj = gzip.open(filepath, "rt")
            self.files.append(fobj)
            raw_event_stream = filter(Event.is_parsed, map(json.loads, fobj))
            iterables.append(map(Event.from_raw, raw_event_stream))
        return stream_events(iterables)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for fobj in self.files:
            fobj.close()

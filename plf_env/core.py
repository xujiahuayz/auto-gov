from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import total_ordering
from typing import Any, Dict

import stringcase


class BaseAction(ABC):
    @abstractmethod
    def execute(self, env: "PLFEnv"):
        pass

    def is_user_action(self):
        return False


class UserAction(BaseAction):
    def __init__(self, user: str, market: str):
        self.user = user
        self.market = market

    def is_user_action(self):
        return True


class ProtocolAction(BaseAction):
    def __init__(self, market: str):
        self.market = market

    def is_protocol_action(self):
        return True


@dataclass
@total_ordering
class Event:
    args: Dict[str, Any]
    event: str
    log_index: int
    transaction_index: int
    transaction_hash: str
    address: str
    block_hash: str
    block_number: int

    @classmethod
    def from_raw(cls, raw_event: Dict[str, Any]):
        if "event" not in raw_event:
            raise ValueError(f"event has not been parsed: {raw_event}")
        snake_raw_event = {
            stringcase.snakecase(key): value for key, value in raw_event.items()
        }
        return Event(**snake_raw_event)

    @classmethod
    def from_string(cls, raw_event: str) -> Event:
        return cls.from_raw(json.loads(raw_event))

    @staticmethod
    def is_parsed(raw_event: str) -> bool:
        return "event" in raw_event

    @property
    def key(self):
        return (self.block_number, self.transaction_index, self.log_index)

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Event) and self.key == o.key

    def __lt__(self, o: object):
        if not isinstance(o, Event):
            raise TypeError(f"cannot compare {o.__class__} and Event")
        return self.key < o.key

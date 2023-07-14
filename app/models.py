from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Callable

import orjson

from pydantic import BaseModel, root_validator


def orjson_dumps(v: Any, *, default: Callable[[Any], Any] | None) -> str:
    return orjson.dumps(v, default=default).decode()


def convert_datetime_to_gmt(dt: datetime) -> str:
    if not dt.tzinfo:
        dt = dt.replace(tzinfo=ZoneInfo("KTC"))

    return dt.strftime("%Y-%m-%dT%H:%M:%S%z")


class SingletonMetaClass(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class RequestCounter(metaclass=SingletonMetaClass):
    from asyncio import Lock

    count = 0
    async_lock = Lock()

    async def increase(self):
        async with self.async_lock:
            self.count += 1

    async def decrease(self):
        async with self.async_lock:
            self.count -= 1


class ORJSONMoel(BaseModel):
    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps
        json_encoders = {datetime: convert_datetime_to_gmt}
        allow_populatioin_by_field_name = True

    @root_validator(skip_on_failure=True)
    def set_null_microseconds(cls, data: dict[str, Any]) -> dict[str, Any]:
        datetime_fields = {
            k: v.replace(microsecond=0) for k, v in data.items() if isinstance(k, datetime)
        }

        return {**data, **datetime_fields}

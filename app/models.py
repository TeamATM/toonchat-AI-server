from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Callable
from threading import Thread

import orjson
import torch

from pydantic import BaseModel, root_validator
from transformers import StoppingCriteria


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


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=None, encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops] if stops else []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True

        return False


class LoadedLLM(metaclass=SingletonMetaClass):
    isLoaded = False
    isRunning = False

    def __call__(cls, model, tokenizer, *args, **kwargs):
        from transformers import TextIteratorStreamer

        cls.model = model
        cls.tokenizer = tokenizer
        cls.isLoaded = True
        cls.streamer = TextIteratorStreamer(
            tokenizer, timeout=10, skip_prompt=True, skip_special_tokens=True
        )
        cls.stopping_criteria = StoppingCriteriaSub()

    def generate(cls, history, x):
        generate_kwargs = dict(
            **cls.tokenizer(
                f"{history}Human: {x}\nRemon:", return_tensors="pt", return_token_type_ids=False
            ).to(0),
            streamer=cls.streamer,
            max_new_tokens=512,
            do_sample=True,
            early_stopping=True,
            eos_token_id=2,
            stopping_criteria=cls.stopping_criteria,
            temperature=0.1,
            # top_p=top_p,
            # top_k=top_k
        )
        thread = Thread(target=cls.model.generate, kwargs=generate_kwargs)
        thread.start()

        return cls.streamer

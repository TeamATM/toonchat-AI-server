import torch
from typing import Callable
from threading import Thread
from transformers import StoppingCriteria

from app.models import SingletonMetaClass
from app.llm.constants import ModelType


class LLMConfig:
    def __init__(
        self,
        model_type: ModelType,
        model_path: str,
        base_model_path: str | None = None,
        load_in_4bit: bool = True,
        stopping_words: list | None = None,
    ) -> None:
        self.model_type = model_type
        self.adaptor_path = model_path
        self.base_model_path = base_model_path
        self.load_in_4bit = load_in_4bit
        self.stopping_words = stopping_words


class BaseLLM:
    def generate(self, history: str, x: str, args: dict = None):
        raise NotImplementedError


class MockLLM(BaseLLM):
    def generate(self, history: str, x: str, args: dict = None):
        import time

        for s in ["This", " is", " a", " mock", " result"]:
            yield s
            time.sleep(1)


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=None, callback: Callable = None):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops] if stops else []
        self.callback = callback

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                if self.callback:
                    self.callback()
                return True

        return False


class LoadedLLM(BaseLLM, metaclass=SingletonMetaClass):
    from transformers import PreTrainedTokenizerBase, PreTrainedModel

    isLoaded = False
    isRunning = False
    do_stop = False

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        stopping_words: list = None,
    ) -> None:
        from transformers import TextIteratorStreamer, StoppingCriteriaList

        self.model = model
        self.tokenizer = tokenizer
        self.isLoaded = True
        self.stopping_criteria = None
        self.streamer = TextIteratorStreamer(
            tokenizer, timeout=10, skip_prompt=True, skip_special_tokens=True
        )

        if stopping_words:
            stop_words_ids = [
                tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
                for stop_word in stopping_words
            ]
            self.stopping_criteria = StoppingCriteriaList(
                [
                    StoppingCriteriaSub(stops=stop_words_ids, callback=self.on_stop_generate),
                    _StopEverythingStoppingCriteria(self),
                ]
            )

    def on_stop_generate(self):
        self.isRunning = False

    def stop_generate(self):
        self.do_stop = True

    def generate(self, history, x, args: dict = None):
        # if self.isRunning:
        #     raise
        generate_kwargs = (
            dict(
                **self.tokenizer(
                    f"{history}\nRemon:", return_tensors="pt", return_token_type_ids=False
                ).to(0),
                max_time=20,  # 최대 생성 시간 (s)
                streamer=self.streamer,
                max_new_tokens=512,
                do_sample=True,
                early_stopping=True,
                eos_token_id=2,
                stopping_criteria=self.stopping_criteria,
                temperature=0.1,
                # top_p=top_p,
                # top_k=top_k
            )
            if not args
            else args
        )
        self.isRunning = True
        self.do_stop = False
        thread = Thread(target=self.model.generate, kwargs=generate_kwargs)

        thread.start()
        return self.streamer


class _StopEverythingStoppingCriteria(StoppingCriteria):
    def __init__(self, loaded_llm: LoadedLLM) -> None:
        self.loaded_llm = loaded_llm

    def __call__(self, input_ids, scores) -> bool:
        if self.loaded_llm.do_stop:
            self.loaded_llm.do_stop = False
            return True

        return False

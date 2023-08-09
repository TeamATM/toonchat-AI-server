import os
from typing import Callable
from threading import Thread

from app.models import SingletonMetaClass


class BaseLLM:
    model = None
    promt_template: str = "Toonchat_v2"

    def generate(self, prompt: str, bot=None, **kwargs):
        raise NotImplementedError


class MockLLM(BaseLLM):
    def generate(self, prompt: str, bot=None, **kwargs):
        import time

        # print(prompt)
        for s in ["This", " is", " a", " mock", " result"]:
            time.sleep(1)
            yield s


if not os.environ["MOCKING"]:
    import torch
    from transformers import StoppingCriteria

    class LoadedLLM(BaseLLM, metaclass=SingletonMetaClass):
        from transformers import PreTrainedTokenizerBase, PreTrainedModel

        isLoaded = False
        isRunning = False
        do_stop = False

        def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerBase,
            promt_template: str,
            stopping_words: list = None,
        ) -> None:
            from transformers import TextIteratorStreamer, StoppingCriteriaList

            self.model = model
            self.tokenizer = tokenizer
            self.promt_template = promt_template
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

        def generate(self, prompt: str, bot=None, **kwargs):
            from peft.peft_model import PeftModel
            from app.llm.utils import set_adapter

            generate_kwargs = dict(
                **self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    return_token_type_ids=False,
                ).to(0),
                max_time=15,  # 최대 생성 시간 (s)
                streamer=self.streamer,
                max_new_tokens=512,
                do_sample=True,
                early_stopping=True,
                eos_token_id=2,
                stopping_criteria=self.stopping_criteria,
                temperature=0.1,
            )
            if kwargs:
                generate_kwargs.update(kwargs)

            self.isRunning = True
            self.do_stop = False

            if bot and isinstance(self.model, PeftModel):
                set_adapter(self.model, bot)

            thread = Thread(target=self.model.generate, kwargs=generate_kwargs)

            thread.start()
            return self.streamer

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

    class _StopEverythingStoppingCriteria(StoppingCriteria):
        def __init__(self, loaded_llm: LoadedLLM) -> None:
            self.loaded_llm = loaded_llm

        def __call__(self, input_ids, scores) -> bool:
            if self.loaded_llm.do_stop:
                self.loaded_llm.do_stop = False
                return True

            return False

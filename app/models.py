from threading import Thread

import torch

from transformers import StoppingCriteria


class SingletonMetaClass(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


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

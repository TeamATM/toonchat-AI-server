import time
from abc import ABCMeta, abstractmethod
import logging
from pathlib import Path
from typing import Callable

from app.utils import log_execution_time
from app.llm.prompter import Prompter
from app.data import PromptData
from app.llm.config import llm_config

logger = logging.getLogger(__name__)


class LLM(metaclass=ABCMeta):
    def __init__(self, prompter: Prompter) -> None:
        self.prompter = prompter

    @abstractmethod
    def load(self, dir_path, **kwargs):
        pass

    @abstractmethod
    def generate(self, data: PromptData, **generation_args) -> str:
        pass


class MockLLM(LLM):
    def load(self, dir_path, **kwargs):
        pass

    def generate(self, data: PromptData, **generation_args):
        return "This is a Mock LLM"


class HuggingfaceLLM(LLM):
    model = None
    tokenizer = None

    def load(self, dir_path, **kwargs) -> LLM:
        self.load_tokenizer(dir_path, kwargs)
        self.load_pretrained_model(dir_path, kwargs)

        adaptor_path = kwargs.get("adaptor_path", None)
        if adaptor_path:
            self.load_peft_model(adaptor_path)

        self.model.eval()
        self.model.config.use_cache = True

    @log_execution_time
    def load_peft_model(self, adaptor_path):
        from peft.peft_model import PeftModel

        logger.info("Loading Adapter to model")
        self.model = PeftModel.from_pretrained(
            self.model, adaptor_path, adapter_name=adaptor_path.split("/")[-1]
        )
        logger.info("Finished to load adapter")

    def load_tokenizer(self, dir_path, kwargs):
        from transformers import AutoTokenizer

        logger.info("Start loading Tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(dir_path)
        self.tokenizer.model_max_length = 4096
        self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("Finished loading Tokenizer")

    @log_execution_time
    def load_pretrained_model(self, dir_path, kwargs):
        from transformers import AutoModelForCausalLM, PreTrainedModel
        import torch

        logger.info("Start loading Model")
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            dir_path,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
            device_map={"": 0},
        )
        logger.info("Finished to load Model")

    @log_execution_time
    def generate(self, data: PromptData, **generation_args):
        start_time = time.time()
        prompt = self.prompter.get_prompt(data)

        encoded_prompt = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            max_length=4096,
            return_tensors="pt",
        )

        token_length = len(encoded_prompt[0])
        logger.info(
            f"Start inference. query: {data.get_chat_history_list()[-1].content}, token_len: {token_length}"
        )

        generation_config = {
            "max_time": 10,
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.3,
            "repetition_penalty": 1.3,
            # early_stopping = True,
            # num_beams = 2,
        }

        if generation_args:
            generation_config.update(generation_args)

        try:
            output = self.model.generate(inputs=encoded_prompt.to(0), **generation_config)
        except Exception as e:
            logger.error("Error occured while generating answer.\n%s", e)
            return ""

        decoded_output = self.tokenizer.batch_decode(output)[0]
        logger.info(decoded_output)
        inference_result = decoded_output[
            len(prompt) : -len(self.tokenizer.eos_token)
            if decoded_output.endswith(self.tokenizer.eos_token)
            else len(decoded_output)
        ]
        inference_time = time.time() - start_time
        logger.info(
            f"Inference finished. result:{inference_result}, tps: {(len(output[0]) - token_length)/inference_time} tokens/s"
        )

        return inference_result

    def set_adapter(self, adapter_name: str):
        from peft.peft_model import PeftModel

        if not isinstance(self.model, PeftModel):
            logger.error("model is not a peft model")
            return

        if self.model.active_adapter == adapter_name:
            return

        if adapter_name not in self.model.peft_config:
            adapter_path = Path(llm_config.get_adapter_path(adapter_name))
            if adapter_path.is_dir():
                try:
                    self.model.load_adapter(adapter_path, adapter_name=adapter_name)
                    self.model.set_adapter(adapter_name)
                except Exception as e:
                    logger.error("Can not load adpater name %s\n%s", adapter_name, e)
                    return


from transformers import StoppingCriteria, PreTrainedTokenizer


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, tokenizer: PreTrainedTokenizer, stops=None, callback: Callable = None):
        self.stops = (
            [tokenizer.encode(stop, return_tensors="pt").to("cuda") for stop in stops]
            if stops
            else []
        )
        self.callback = callback

    def __call__(self, input_ids, scores):
        import torch

        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                if self.callback:
                    self.callback()
                return True

        return False

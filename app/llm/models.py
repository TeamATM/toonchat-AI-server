import time
from abc import ABCMeta, abstractmethod
import logging

from app.utils import log_execution_time
from app.llm.prompter import Prompter
from app.data import PromptData

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

    @log_execution_time
    def load_peft_model(self, adaptor_path):
        from peft import PeftModel

        logger.info("Loading Adapter to model")
        self.model = PeftModel.from_pretrained(
            self.model, adaptor_path, adapter_name=adaptor_path.split("/")[-1]
        )
        logger.info("Finished to load adapter")

    def load_tokenizer(self, dir_path, kwargs):
        from transformers import AutoTokenizer

        logger.info("Start loading Tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            dir_path, use_fast=kwargs.get("use_fast", True)
        )
        logger.info("Finished loading Tokenizer")

    @log_execution_time
    def load_pretrained_model(self, dir_path, kwargs):
        from transformers import AutoModelForCausalLM, PreTrainedModel

        logger.info("Start loading Model")
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            dir_path,
            quantization_config=kwargs.get("quantization_config", None),
            device_map={"": 0},
        )
        logger.info("Finished to load Model")

    @log_execution_time
    def generate(self, data: PromptData, **generation_args):
        import torch

        start_time = time.time()
        prompt = self.prompter.get_prompt(data)

        encoded_prompt = self.tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)

        token_length = len(encoded_prompt["input_ids"][0])
        logger.info(
            f"Start inference. query: {data.get_chat_history_list()[-1].content}, token_len: {token_length}"
        )

        generate_kwargs = dict(
            **encoded_prompt.to(0),
            max_time=15,
            max_new_tokens=256,
            do_sample=True,
            early_stopping=True,
            eos_token_id=2,
            temperature=0.1,
        )
        if generation_args:
            generate_kwargs.update(generation_args)

        try:
            with torch.no_grad():
                output = self.model.generate(**generate_kwargs)
        except Exception:
            logger.error("Error occured while generating answer.")
            return ""

        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        inference_result = decoded_output[len(prompt) :]
        inference_time = time.time() - start_time
        logger.info(
            f"Inference finished. result:{inference_result}, tps: {(len(output[0]) - token_length)/inference_time} tokens/s"
        )

        return inference_result

import time
from abc import ABCMeta, abstractmethod
import logging
from pathlib import Path

from app.utils import log_execution_time, path_concat
from app.message_queue.data import PromptData
from app.llm.prompter import Prompter
from app.llm.constants import ModelType
from app.llm.config import llm_config, generation_config
from app.llm.tokenizer import load_tokenizer, tokenizer_encode, tokenizer_decode

logger = logging.getLogger(__name__)


class LLM(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def generate(self, data: PromptData, **generation_kwargs):
        pass


class MockLLM(LLM):
    def __init__(self, **kwargs) -> None:
        pass

    def generate(self, data: PromptData, **generation_kwargs):
        return "This is a Mock LLM"


class HuggingfaceLLM(LLM):
    model = None

    def __init__(self, **kwargs) -> None:
        self.tokenizer = load_tokenizer(kwargs.get("pretrained_model_name_or_path"), kwargs)
        prompter_class = kwargs.pop("prompter_class")
        self.prompter: Prompter = prompter_class(tokenizer=self.tokenizer)
        self.load_pretrained_model(kwargs.get("pretrained_model_name_or_path"), kwargs)

        if kwargs.pop("model_type") == ModelType.LoRA:
            self.load_peft_model(path_concat(kwargs.pop("adapter_dir"), kwargs.pop("adapter_name")))

        self.model.eval()
        self.model.config.use_cache = True

    @log_execution_time
    def load_pretrained_model(self, pretrained_model_name_or_path, kwargs: dict):
        from transformers import AutoModelForCausalLM, PreTrainedModel
        import torch

        logger.info("Start loading Model from path: %s", pretrained_model_name_or_path)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.bfloat16,
            load_in_4bit=kwargs.pop("load_in_4bit", True),
            device_map={"": 0},
        )
        logger.info("Finished to load Model")

    @log_execution_time
    def load_peft_model(self, adaptor_path):
        from peft.peft_model import PeftModel

        logger.info("Loading Adapter to model from path: %s", adaptor_path)
        self.model = PeftModel.from_pretrained(self.model, adaptor_path, adapter_name="default")
        logger.info("Finished to load adapter")

    def generate(self, data: PromptData, **generation_kwargs):
        start_time = time.time()
        prompt = self.prompter.get_prompt(data)
        encoded_prompt = tokenizer_encode(self.tokenizer, prompt)
        decoded_prompt = tokenizer_decode(self.tokenizer, encoded_prompt)

        token_length = len(encoded_prompt[0])
        logger.info(
            f"Start inference. query: {data.get_chat_history_list()[-1].content}, token_len: {token_length}"
        )

        try:
            output = self.model.generate(
                inputs=encoded_prompt.to(0), **{**generation_config, **generation_kwargs}
            )
        except Exception as e:
            logger.error("Error occured while generating answer.\n%s", e)
            return ""

        decoded_output = tokenizer_decode(self.tokenizer, output)
        inference_result = self.extract_answer(decoded_output, decoded_prompt)
        inference_time = time.time() - start_time
        logger.info(
            f"Inference finished. result:{inference_result}, tps: {(len(output[0]) - token_length)/inference_time} tokens/s"
        )

        return inference_result.split(".")[0] + "."

    def extract_answer(self, decoded_output: str, prompt: str):
        return decoded_output[
            len(prompt) : -len(self.tokenizer.eos_token)
            if decoded_output.endswith(self.tokenizer.eos_token)
            else len(decoded_output)
        ]

    def set_adapter(self, adapter_name: str):
        from peft.peft_model import PeftModel

        if not isinstance(self.model, PeftModel):
            logger.error("model is not a peft model")
            return

        if self.model.active_adapter == adapter_name:
            return

        adapter = "default"
        if adapter_name not in self.model.peft_config:
            adapter_path = Path(llm_config.get_adapter_path(adapter_name))
            if adapter_path.is_dir():
                try:
                    logger.info("trying to load adapter with path: %s", adapter_path)
                    self.model.load_adapter(adapter_path, adapter_name=adapter_name)
                    logger.info("load adapter: %s", adapter_name)
                    adapter = adapter_name
                except Exception as e:
                    logger.error("Can not load adpater name %s\n%s", adapter_name, e)
        logger.info("set adapter: %s", adapter)
        self.model.set_adapter(adapter)

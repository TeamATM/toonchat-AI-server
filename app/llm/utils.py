import logging

from app.utils import is_production
from pathlib import Path

if not is_production():
    from app.llm.models import MockLLM
else:
    from peft import PeftModel, PeftConfig
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
    )
    from app.llm.models import LoadedLLM
    from app.llm.config import llm_config, bnb_config


from app.llm.models import BaseLLM
from app.llm.constants import ModelType

logger = logging.getLogger(__name__)


def load_model() -> BaseLLM:
    logger.info("Start loading Model")
    if not is_production():
        model = MockLLM()
        logger.info("Finished to load Model")
        return model

    try:
        if llm_config.model_type == ModelType.LoRA and not llm_config.base_model_path:
            config = PeftConfig.from_pretrained(llm_config.adapter_path)
            base_model_id = config.base_model_name_or_path
        else:
            base_model_id = llm_config.base_model_path

        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, quantization_config=bnb_config, device_map={"": 0}
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id, use_fast=True, model_max_length=4096
        )

    except ValueError as e:
        raise e

    logger.info("Finished to load Model")
    if llm_config.model_type == ModelType.LoRA:
        logger.info("Loading Adapter to model")
        model = PeftModel.from_pretrained(
            model, llm_config.get_adapter_path(), adapter_name=llm_config.adapter_name
        )
        logger.info("Finished to load adapter")

    return LoadedLLM(
        model,
        tokenizer,
        promt_template=llm_config.prompt_template,
        stopping_words=llm_config.stopping_words,
    )


def set_adapter(model, adapter_name: str):
    if not isinstance(model, PeftModel):
        logger.error("model is not a peft model")
        return

    if model.active_adapter == adapter_name:
        return

    if adapter_name not in model.peft_config:
        adapter_path = Path(llm_config.get_adapter_path(adapter_name))
        if adapter_path.is_dir():
            try:
                model.load_adapter(
                    llm_config.get_adapter_path(adapter_name), adapter_name=adapter_name
                )
                model.set_adapter(adapter_name)
            except Exception as e:
                logger.error("Can not load adpater name %s\n%s", adapter_name, e)
                return

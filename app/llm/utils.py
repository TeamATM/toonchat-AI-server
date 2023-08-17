from app.utils import is_production
from pathlib import Path

if not is_production():
    from app.llm.models import MockLLM
else:
    from torch import bfloat16
    from peft import PeftModel, PeftConfig
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )
    from app.llm.models import LoadedLLM


from app.llm.models import BaseLLM
from app.llm.config import llm_config
from app.llm.constants import ModelType


def load_model() -> BaseLLM:
    if not is_production():
        model = MockLLM()
        return model
    else:
        bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=bfloat16,
            )
            if llm_config.load_in_4bit
            else None
        )

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

    if llm_config.model_type == ModelType.LoRA:
        model = PeftModel.from_pretrained(
            model, llm_config.get_adapter_path(), adapter_name=llm_config.adapter_name
        )

    return LoadedLLM(
        model,
        tokenizer,
        promt_template=llm_config.prompt_template,
        stopping_words=llm_config.stopping_words,
    )


def set_adapter(model, adapter_name: str):
    if not isinstance(model, PeftModel):
        print("model is not a peft model")
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
                print(f"Can not load adpater name {adapter_name}\n{e}")
                return

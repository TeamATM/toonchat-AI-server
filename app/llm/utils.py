from os import environ
from os.path import join
from peft import PeftModel, PeftConfig
from torch import bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from app.llm.models import LoadedLLM, LLMConfig, MockLLM, BaseLLM
from app.llm.constants import ModelType


def load_lora(model_config: LLMConfig):
    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=bfloat16,
        )
        if model_config.load_in_4bit
        else None
    )

    try:
        if model_config.model_type == ModelType.LoRA and not model_config.base_model_path:
            config = PeftConfig.from_pretrained(model_config.adaptor_path)
            base_model_id = config.base_model_name_or_path
        else:
            base_model_id = model_config.base_model_path

        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, quantization_config=bnb_config, device_map={"": 0}
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        if model_config.model_type == ModelType.LoRA:
            model = PeftModel.from_pretrained(model, model_config.adaptor_path)

    except ValueError as e:
        raise e

    model.eval()

    loaded_model: LoadedLLM = LoadedLLM(
        model=model, tokenizer=tokenizer, stopping_words=model_config.stopping_words
    )

    return loaded_model


def load_model() -> BaseLLM:
    if "MOCKING" in environ and environ["MOCKING"]:
        model = MockLLM()
    else:
        model = load_lora(
            LLMConfig(
                ModelType.LoRA,
                join("models", "checkpoint-2200"),
                join("models", "beomi", "KoAlpaca-Polyglot-5.8B"),
                stopping_words=["\n\n", "\nHuman:"],
            ),
        )

    return model

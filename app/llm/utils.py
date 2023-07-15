from peft import PeftModel, PeftConfig
from torch import bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from app.llm.models import LoadedLLM, LLMConfig
from app.llm.constants import ModelType

loaded_model_dict = {}


def load_lora(model_config: LLMConfig, tag: str = None):
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

        tag = tag if tag else base_model_id
        if tag in loaded_model_dict:
            return loaded_model_dict[tag]

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
    loaded_model_dict[tag] = loaded_model

    return loaded_model

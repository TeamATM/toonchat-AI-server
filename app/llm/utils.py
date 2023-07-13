from peft import PeftModel, PeftConfig
from torch import bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from app.models import LoadedLLM


def load_lora(peft_model_id: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=bfloat16,
    )

    try:
        config = PeftConfig.from_pretrained(peft_model_id)
        base_model_id = config.base_model_name_or_path

        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, quantization_config=bnb_config, device_map={"": 0}
        )
    except ValueError as e:
        raise e

    model = PeftModel.from_pretrained(model, peft_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    model.eval()

    return LoadedLLM(model=model, tokenizer=tokenizer)

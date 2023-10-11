from os import environ


if environ["MOCKING"]:
    from app.llm.models import MockLLM
else:
    from torch import bfloat16
    from peft import PeftModel, PeftConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from app.llm.models import LoadedLLM


from app.llm.models import BaseLLM
from app.llm.config import LLMConfig
from app.llm.constants import ModelType


def load_model(llm_config: LLMConfig) -> BaseLLM:
    if environ["MOCKING"]:
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
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    except ValueError as e:
        raise e

    if llm_config.model_type == ModelType.LoRA:
        model = PeftModel.from_pretrained(model, llm_config.adapter_path)

    return LoadedLLM(
        model,
        tokenizer,
        promt_template=llm_config.prompt_template,
        stopping_words=llm_config.stopping_words,
    )


# def load_prompt(fname):
#     file_path = Path(f"app/prompts/{fname}.yaml")
#     if not file_path.exists():
#         print_red("Template is Empty!")
#         return ""

#     with open(file_path, "r", encoding="utf-8") as f:
#         data = yaml.safe_load(f)
#         output = ""
#         if "context" in data:
#             output += data["context"]

#         replacements = {
#             "<|user|>": data["user"],
#             "<|sep|>": data["sep"],
#             "<|bot|>": data["bot"],
#         }

#         output += replace_all(
#             data["turn_template"].split("<|bot-message|>")[0], replacements
#         ).strip()
#         return {"prompt": output, "sep": data["sep"]}

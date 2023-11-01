from pathlib import Path
from pydantic import root_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.llm.constants import ModelType
from app.utils import is_local, path_concat, get_profile


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=f"env/.env.{get_profile().value}", env_file_encoding="utf-8", extra="allow"
    )

    model_type: ModelType
    pretrained_model_name_or_path: str
    adapter_dir: str | None = None
    adapter_name: str | None = None
    prompt_template: str = "Toonchat_v2.1"
    load_in_4bit: bool = True
    model_max_length: int

    @root_validator(pre=True)
    def a(cls, values: dict):
        if is_local():
            return values

        pretrained_model_name_or_path = values.get("pretrained_model_name_or_path")
        if not pretrained_model_name_or_path or not Path(pretrained_model_name_or_path).is_dir():
            raise FileNotFoundError("Can not find model file")

        adapter_path = path_concat(values.get("adapter_dir"), values.get("adapter_name"))
        if values.get("model_type") == ModelType.LoRA and (
            not adapter_path or not Path(adapter_path).is_dir()
        ):
            raise FileNotFoundError("Can not find lora file")

        return values

    def get_adapter_path(self, adapter_name: str = None):
        return path_concat(self.adapter_dir, adapter_name if adapter_name else self.adapter_name)


llm_config = LLMConfig()
generation_config = {
    "max_time": 10,
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.3,
    "repetition_penalty": 1.3,
    # early_stopping = True,
    # num_beams = 2,
}

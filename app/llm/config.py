from os import path
from pathlib import Path
from pydantic import root_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.llm.constants import ModelType
from app.utils import is_production


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    model_type: ModelType
    base_model_path: str
    adapter_path: str | None = None
    adapter_name: str | None = None
    prompt_template: str = "Toonchat_v2"
    load_in_4bit: bool = True
    stopping_words: list | str = None

    @root_validator(pre=True)
    def a(cls, values: dict):
        if not is_production():
            return values

        base_model_path = values.get("base_model_path")
        if not base_model_path or not Path(base_model_path).is_dir():
            raise FileNotFoundError("Can not find model file")

        adapter_path = path.join(values.get("adapter_path"), values.get("adapter_name"))
        if (values.get("model_type") == ModelType.LoRA and not adapter_path) or not Path(
            adapter_path
        ).is_dir():
            raise FileNotFoundError("Can not find lora file")

        sw = values.get("stopping_words")
        if sw and isinstance(sw, str):
            values["stopping_words"] = sw.split(", ")

        return values

    def get_adapter_path(self, adapter_name: str = None):
        return path.join(self.adapter_path, adapter_name if adapter_name else self.adapter_name)


llm_config = LLMConfig()

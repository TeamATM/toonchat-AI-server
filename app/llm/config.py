from pathlib import Path
from pydantic import root_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.llm.constants import ModelType


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env.llm", env_file_encoding="utf-8", extra="allow")

    model_type: ModelType
    base_model_path: str
    adapter_path: str | None = None
    prompt_fname: str = None
    load_in_4bit: bool = True
    stopping_words: list | None = None

    @root_validator(pre=True)
    def a(cls, values: dict):
        base_model_path = values.get("base_model_path")
        if not base_model_path or not Path(base_model_path).is_dir():
            raise FileNotFoundError

        adapter_path = values.get("adapter_path")
        if (values.get("model_type") == ModelType.LoRA and not adapter_path) or not Path(
            adapter_path
        ).is_dir():
            raise FileNotFoundError

        return values


llm_config = LLMConfig()
print(llm_config)

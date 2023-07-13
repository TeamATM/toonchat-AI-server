from os.path import join
from os import environ
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.constants import Environment
from app.llm.constants import LLMModel


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=join(environ["ROOT_DIR"], ".env"),
        env_file_encoding="utf-8",
        use_enum_values=True,
    )

    REDIS_URL: str

    SITE_DOMAIN: str

    ENVIRONMENT: Environment

    CORS_ORIGINS: list[str]
    CORS_HEADERS: list[str]

    APP_VERSION: str

    PEFT_MODEL_DIR: str | LLMModel

    @field_validator("PEFT_MODEL_DIR")
    def validate_model_dir(cls, v):
        assert v in LLMModel._member_map_, "invalid model name"
        return LLMModel._member_map_.get(v)


settings = Config()

app_configs: dict[str, Any] = {}
app_configs["root_path"] = f"/v{settings.APP_VERSION}"

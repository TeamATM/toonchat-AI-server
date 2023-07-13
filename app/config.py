from typing import Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.constants import Environment
from os.path import join, dirname


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=join(dirname(dirname(__file__)), ".env"), env_file_encoding="utf-8"
    )

    REDIS_URL: str

    SITE_DOMAIN: str

    ENVIRONMENT: Environment

    CORS_ORIGINS: list[str]
    CORS_HEADERS: list[str]

    APP_VERSION: str


settings = Config()

app_configs: dict[str, Any] = {}
app_configs["root_path"] = f"/v{settings.APP_VERSION}"

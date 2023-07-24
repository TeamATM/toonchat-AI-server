from pydantic_settings import BaseSettings, SettingsConfigDict


class CeleryConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.celery", env_file_encoding="utf-8", extra="allow"
    )

    BROKER_URI: str
    BACKEND_URI: str | None


config = CeleryConfig()

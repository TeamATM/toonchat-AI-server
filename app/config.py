from pydantic_settings import BaseSettings, SettingsConfigDict

from app.utils import get_profile


class CeleryConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=f"env/.env.{get_profile().value}", env_file_encoding="utf-8", extra="allow"
    )

    broker_url: str
    task_default_queue: str
    task_default_exchange: str
    task_default_routing_key: str

    def to_dict(self):
        return self.model_dump()


celeryConfig = CeleryConfig()

from os import environ
from os.path import dirname, join

environ["ROOT_DIR"] = dirname(dirname(__file__))

from app.llm.utils import load_lora
from app.llm.constants import ModelType
from app.llm.models import LLMConfig

loaded_model = load_lora(
    LLMConfig(
        ModelType.LoRA,
        join(environ["ROOT_DIR"], "models", "checkpoint-2200"),
        join(environ["ROOT_DIR"], "models", "beomi", "KoAlpaca-Polyglot-5.8B"),
        stopping_words=["\n\n", "\nHuman:"],
    ),
    tag="Remon-2200",
)

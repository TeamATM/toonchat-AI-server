from enum import Enum
from os.path import join
from os import environ


class LLMModel(str, Enum):
    POLYGLOT_5_8 = join(environ["ROOT_DIR"], "models", "beomi", "KoAlpaca-Polyglot-5.8B")
    POLYGLOT_12_8 = join(environ["ROOT_DIR"], "models", "beomi", "KoAlpaca-Polyglot-12.8B")

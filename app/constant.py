from enum import Enum


class Profile(str, Enum):
    PRODUCTION = "production"
    DEVELOP = "develop"
    LOCAL = "local"

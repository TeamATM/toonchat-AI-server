from os import environ


# Replace multiple string pairs in a string
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)

    return text


def print_red(text):
    print(f"\033[31m{text}\033[0m")


def get_profile():
    return environ.get("PROFILE", "local")


def is_production():
    return get_profile == "production"

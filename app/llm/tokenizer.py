import logging

logger = logging.getLogger(__name__)


def load_tokenizer(pretrained_model_name_or_path, kwargs: dict):
    from transformers import AutoTokenizer

    logger.info("Start loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.model_max_length = kwargs.pop("model_max_length", 4096)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Finished loading Tokenizer")

    return tokenizer


def tokenizer_encode(tokenizer, prompt):
    return tokenizer.encode(
        prompt,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )


def tokenizer_decode(tokenizer, sequences):
    decoded_output = tokenizer.batch_decode(sequences)[0]
    logger.info(decoded_output)
    return decoded_output

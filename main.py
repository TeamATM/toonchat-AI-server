from os import chdir
from os.path import dirname, join
import asyncio

chdir(dirname(__file__))

from app.config import settings
from app.redis.config import Redis
from app.redis.consumer import StreamConsumer

redis = Redis()

if settings.ENVIRONMENT.is_testing or settings.ENVIRONMENT.is_deployed:
    from app.llm.utils import load_lora
    from app.llm.constants import ModelType
    from app.llm.models import LLMConfig

    loaded_model = load_lora(
        LLMConfig(
            ModelType.LoRA,
            join("models", "checkpoint-2200"),
            join("models", "beomi", "KoAlpaca-Polyglot-5.8B"),
            stopping_words=["\n\n", "\nHuman:"],
        ),
        tag="Remon-2200",
    )
else:
    from app.llm.models import MockLLM

    loaded_model = MockLLM()


def handle_stream(streamer):
    for stream in streamer:
        print(stream)


async def main():
    redis_client = redis.create_connection()
    consumer = StreamConsumer(redis_client)

    await consumer.create_consumer(redis.REQUEST_STREAM, redis.CONSUMER_GROUP, "AI1")

    while True:
        response = await consumer.consume_stream(
            steram_channel=redis.REQUEST_STREAM,
            group_name=redis.CONSUMER_GROUP,
            consumer_name="AI1",
            count=1,
            block=0,
        )
        if response:
            processed_messages = []
            for message_id, _, v in response:
                streamer = loaded_model.generate(history="", x=v)
                handle_stream(streamer)
                processed_messages.append(message_id)
            consumer.ack_message(redis.REQUEST_STREAM, redis.CONSUMER_GROUP, *processed_messages)

        await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(main())

from os import chdir
from os.path import dirname, join
import asyncio

chdir(dirname(__file__))

from app.config import settings
from app.redis.config import Redis
from app.redis.consumer import StreamConsumer
from app.redis.models import Message
from app.redis.history import Cache

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
    result = []

    for stream in streamer:
        result.append(stream)
        print(stream)
    print()

    return result


async def main():
    redis_client = redis.create_connection()
    consumer = StreamConsumer(redis_client)
    cache = Cache(redis_client)

    await consumer.create_consumer(redis.REQUEST_STREAM, redis.CONSUMER_GROUP, "AI1")

    while True:
        response = await consumer.consume_stream(
            steram_channel=redis.REQUEST_STREAM,
            group_name=redis.CONSUMER_GROUP,
            consumer_name="AI1",
            count=2,
            block=0,
        )
        if response:
            processed_messages = []
            for message_id, k, v in response:
                message = Message(msg=v)

                await cache.add_message_to_cache(
                    key=k, source="Human", message_data=message.model_dump()
                )
                chat_history = await cache.get_chat_history(k)
                history = "\n".join([history["msg"] for history in chat_history["messages"][-4:]])

                streamer = loaded_model.generate(history=history, x=v)
                generated = "".join(handle_stream(streamer))
                message = Message(msg=generated)

                await cache.add_message_to_cache(
                    key=k, source="Remon", message_data=message.model_dump()
                )
                processed_messages.append(message_id)

            await consumer.ack_message(
                redis.REQUEST_STREAM, redis.CONSUMER_GROUP, processed_messages
            )

        await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(main())

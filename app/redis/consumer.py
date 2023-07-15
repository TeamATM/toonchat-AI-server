from redis.asyncio import Redis
from redis.typing import KeyT, StreamIdT
from redis.exceptions import ResponseError


class StreamConsumer:
    def __init__(self, redis_client: Redis) -> None:
        self.redis_client = redis_client

    async def _create_consumer_group(self, stream_name: str, group_name: str):
        try:
            await self.redis_client.xgroup_create(stream_name, group_name, mkstream=True)
        except ResponseError:
            pass

    async def create_consumer(self, stream_name: str, group_name: str, consumer_name: str):
        await self._create_consumer_group(stream_name, group_name)

        try:
            await self.redis_client.xgroup_createconsumer(stream_name, group_name, consumer_name)
        except ResponseError:
            pass

    async def consume_stream(
        self, group_name: str, consumer_name: str, count: int, block: int, steram_channel: KeyT
    ):
        response = await self.redis_client.xreadgroup(
            groupname=group_name,
            consumername=consumer_name,
            streams={steram_channel: ">"},
            count=count,
            block=block,
        )
        if response:
            items = []
            for _, messages in response:
                for message_id, message in messages:
                    k, v = map(bytes.decode, message.popitem())
                    items.append((message_id, k, v))
            return items
        return response

    async def delete_message(self, stream_channel: KeyT, message_id: StreamIdT):
        await self.redis_client.xdel(stream_channel, message_id)

    async def ack_message(self, stream_name: str, group_name: str, *ids: StreamIdT):
        await self.redis_client.xack(stream_name, group_name, ids)

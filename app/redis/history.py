from redis.asyncio import Redis
from redis.commands.json.path import Path
from redis.exceptions import ResponseError


class Cache:
    def __init__(self, redis_client: Redis) -> None:
        self.redis_client = redis_client

    async def get_chat_history(self, key: str):
        data = await self.redis_client.json().get(key, Path.root_path())
        return data

    async def add_message_to_cache(self, key: str, source: str, message_data: dict):
        message_data["msg"] = f"{source}: {message_data['msg']}"
        try:
            await self.redis_client.json().arrappend(key, Path(".messages"), message_data)
        except ResponseError:
            await self.redis_client.json().set(key, "$", {"messages": [message_data]}, nx=True)

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional


@dataclass
class Message:
    messageId: str
    replyMessageId: Optional[str]
    createdAt: datetime
    content: str
    fromUser: bool

    def is_user(self) -> bool:
        return self.fromUser


@dataclass
class History:
    _id: str
    userId: str
    characterId: int
    messages: list[Message]


@dataclass
class GenerationArgs:
    temperature: float
    repetition_penalty: float


@dataclass
class PromptData:
    persona: str
    reference: list[str]
    history: History
    generationArgs: dict
    greetingMessage: str = None

    def get_user_id(self) -> str:
        return self.history.userId

    def get_character_id(self) -> int:
        return self.history.characterId

    def get_greeting_message(self) -> list[dict] | None:
        if not self.greetingMessage:
            return None
        try:
            data = json.loads(self.greetingMessage)
            assert isinstance(data, list)
            for chat in data:
                assert isinstance(chat, dict) and "role" in chat and "content" in chat
            return data
        except (AssertionError, Exception):
            return [{"role": "assistant", "content": self.greetingMessage}]

    def get_persona(self) -> str:
        return " ".join(self.persona) if isinstance(self.persona, list) else self.persona

    def get_reference(self) -> str:
        if isinstance(self.reference, list):
            return " ".join(self.reference)
        else:
            return self.reference

    def get_chat_history_list(self) -> list[Message]:
        return self.history.messages

    def get_generation_args(self):
        return self.generationArgs

    def build_return_message(self, message_id: str, content: str):
        return MessageToMq(
            messageId=message_id,
            userId=self.history.userId,
            characterId=self.history.characterId,
            createdAt=datetime.now(timezone.utc).isoformat(),
            content=content,
            fromUser=False,
        )


@dataclass
class MessageToMq:
    messageId: str
    userId: str
    characterId: int
    createdAt: datetime
    content: str
    fromUser: bool = False

    def to_dict(self):
        return asdict(self)

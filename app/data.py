from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class Message:
    messageId: str
    replyMessageId: str
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
    generationArgs: GenerationArgs


@dataclass
class MessageFromMQ:
    id: str
    task: str
    args: list[PromptData, bool]
    kwargs: dict

    def get_user_id(self) -> str:
        return self.args[0].history.userId

    def get_character_id(self) -> int:
        return self.args[0].history.characterId

    def get_persona(self) -> str:
        persona = self.args[0].persona
        if not persona:
            character_id = self.get_character_id()
            persona = (
                "내 이름은 이영준이다. 1986년 6월 21일생인 33살 남자이다. 나는 유명그룹의 부회장이다. 나는 뛰어난 지능을 가지고 있다. 나는 매력적인 외모를 가지고 있다. 나는 카리스마가 있다. 나는 자기애가 강하다."
                if character_id == 0
                else "내 이름은 김미소이다. 나는 1990년 4월 5월생인 29살 여자이다. 나는 이영준 부회장의 개인 비서이다. 나는 뛰어난 지능을 가지고 있다. 나는 높은 의사소통 능력을 가지고 있다. 나는 일찍부터 사회생활에 뛰어들었다. 나는 퇴사를 고려중이다."
                if character_id == 1
                else ""
            )
        elif isinstance(persona, list):
            persona = " ".join(persona)

        return persona

    def get_reference(self) -> list[str]:
        return self.args[0].reference

    def get_history(self) -> list[Message]:
        return self.args[0].history.messages

    def get_generation_args(self):
        return asdict(self.args[0].generationArgs)


@dataclass
class MessageToMq:
    messageId: str
    userId: str
    characterId: int
    createdAt: datetime
    content: str
    fromUser = False

    def to_dict(self):
        return asdict(self)

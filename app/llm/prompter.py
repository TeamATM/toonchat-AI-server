from abc import ABCMeta, abstractmethod
from app.data import PromptData


class Prompter(metaclass=ABCMeta):
    @abstractmethod
    def get_prompt(messages: PromptData) -> str:
        pass


class MockPrompter(Prompter):
    def get_prompt(messages: PromptData):
        return "Mock Prompt"


class HuggingfacePrompter(Prompter):
    def __init__(self) -> None:
        pass


class ToonchatV21Prompter(HuggingfacePrompter):
    def get_prompt(messages: PromptData):
        tmp = []
        tmp.append(
            """아래는 사용자와의 이전 대화 내용들과 캐릭터에 대한 정보 또는 대화에 필요한 추가 컨텍스트를 제공하는 입력이 짝을 이루는 예제입니다. 요청을 적절히 완료하는 응답을 작성하세요.

### 명령어:"""
        )

        for message in messages.get_chat_history_list()[:-1]:
            tmp.append(f"{'User' if message.fromUser else 'Assistant'}: {message.content}")

        tmp.append("\n### 입력:")
        tmp.append(f"System: {messages.get_persona()}")
        tmp.append(f"User: {messages.get_chat_history_list()[-1].content}")
        tmp.append("\n### 응답:\n")
        return "\n".join(tmp)


class ToonchatV23Prompter(HuggingfacePrompter):
    def get_prompt(self, data: PromptData):
        tmp = []
        tmp.append(f"### Persona: {data.get_persona()}")
        tmp.append(f"### Reference: {' '.join(data.get_reference())}")
        for message in data.get_chat_history_list():
            tmp.append(
                f"### {'Human' if message.fromUser else 'Assistant'}: {message.content}{'' if message.fromUser else '</s>'}"
            )
        tmp.append("### Assistant: ")
        return " ".join(tmp)

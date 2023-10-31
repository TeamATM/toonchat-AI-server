from abc import ABCMeta, abstractmethod
from app.data import PromptData
import logging

logger = logging.getLogger(__name__)


class Prompter(metaclass=ABCMeta):
    def __init__(self, **kwargs) -> None:
        logger.info("Selected Prompter: %s", self.__class__.__name__)

    @abstractmethod
    def get_prompt(self, messages: PromptData) -> str:
        pass


class MockPrompter(Prompter):
    def get_prompt(self, messages: PromptData):
        return "Mock Prompt"


class ToonchatV21Prompter(Prompter):
    def get_prompt(self, messages: PromptData):
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


class ToonchatV23Prompter(Prompter):
    def get_prompt(self, messages: PromptData):
        tmp = []
        tmp.append(f"### Persona: {messages.get_persona()}")
        tmp.append(f"### Reference: {messages.get_reference()}")
        for message in messages.get_chat_history_list():
            tmp.append(
                f"### {'Human' if message.fromUser else 'Assistant'}: {message.content}{'' if message.fromUser else '</s>'}"
            )
        tmp.append("### Assistant: ")
        return " ".join(tmp)


class ChatMlPrompter(Prompter):
    def get_prompt(self, messages: PromptData) -> str:
        tmp = []
        tmp.append(
            "<|im_start|>system\n당신은 주어진 캐릭터의 말투, 성격을 모방하여 대답하는 챗봇입니다. 유저의 질문에 대하여 캐릭터의 특징을 최대한 살려 대답하세요.<|im_end|>"
        )
        if len(messages.get_persona()) > 1:
            tmp.append(
                f"<|im_start|>system\n아래는 당신의 페르소나입니다. 유저의 질문에 대하여 다음의 페르소나를 고려하여 적절한 대답을 완성하세요. {messages.get_persona()}<|im_end|>"
            )
        if messages.get_reference():
            tmp.append(
                f"<|im_start|>system\n다음은 유저의 질문과 관련된 소설의 내용입니다. 유저의 질문에 대해 다음의 내용을 바탕으로 적절한 대답을 완성하세요. {messages.get_reference()}<|im_end|>"
            )
        for i, message in enumerate(messages.get_chat_history_list()):
            if i == 0 and not message.fromUser:
                continue
            # assert i == 0 or (i != 0 and message.fromUser != messages.get_chat_history_list()[i-1].fromUser)
            tmp.append(
                f"<|im_start|>{'user' if message.fromUser else 'assistant'}\n{message.content}<|im_end|>"
            )
        tmp.append("<|im_start|>assistant\n")
        return "\n".join(tmp)


class HuggingfaceTokenizerPrompter(Prompter):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        from transformers import PreTrainedTokenizer

        self.tokenizer: PreTrainedTokenizer = kwargs.pop("tokenizer")

    def get_prompt(self, messages: PromptData) -> str:
        chat_messages = [
            {"role": "user" if message.fromUser else "assistant", "content": message.content}
            for message in messages.get_chat_history_list()
        ]
        return self.tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )

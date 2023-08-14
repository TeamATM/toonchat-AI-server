from typing import List, Dict
from enum import auto, IntEnum
import dataclasses


class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_TWO = auto()
    DOLLY = auto()
    GUANACO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The names of two roles
    roles: List[str] = (("USER", "ASSISTANT"),)
    # All messages. Each item is (role, message).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.DOLLY
    sep: str = "\n"
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.GUANACO:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            message_len = len(self.messages)
            instruct_len = message_len - 2

            # there is a system message
            if message_len >= 3 and self.messages[-3][0] == self.roles[2]:
                instruct_len -= 1

            if instruct_len == 0:
                ret += "\n"
            for i, (role, message) in enumerate(self.messages[:instruct_len]):
                if message:
                    ret += role + ": " + message + seps[i + 1 == instruct_len]
                elif i + 1 == instruct_len:
                    ret += "\n\n"

            ret += "### 입력:\n"
            for i, (role, message) in enumerate(self.messages[instruct_len:-1]):
                if message:
                    ret += role + ": " + message
                    ret += seps[instruct_len + i + 2 == message_len]

            ret += "### 응답:\n"
            if self.messages[-1][1]:
                ret += self.messages[-1][1]

            return ret

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def extend_messages(self, messages: List[List[str]]):
        """Extend messages."""
        self.messages.extend(messages)

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        ret = [{"role": "system", "content": system_prompt}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert template.name not in conv_templates, f"{template.name} has been registered."

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


register_conv_template(
    Conversation(
        name="Toonchat_v1.1",
        system_message="아래는 작업을 설명하는 명령어입니다. 요청을 적절히 완료하는 응답을 작성하세요.\n\n",
        roles=("### 명령어", "### 응답"),
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.DOLLY,
        sep="\n\n",
        sep2="</s>",
    )
)

register_conv_template(
    Conversation(
        name="Toonchat_v2",
        system_message="""아래는 작업을 설명하는 명령어와 추가 컨텍스트를 제공하는 입력이 짝을 이루는 예제입니다. 요청을 적절히 완료하는 응답을 작성하세요.

### 명령어:
""",
        roles=("User", "Assistant", "System"),
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.GUANACO,
        sep=" ",
        sep2="\n\n",
    )
)

register_conv_template(
    Conversation(
        name="Toonchat_v2.1",
        system_message="""아래는 사용자와의 이전 대화 내용들과 캐릭터에 대한 정보 또는 대화에 필요한 추가 컨텍스트를 제공하는 입력이 짝을 이루는 예제입니다. 요청을 적절히 완료하는 응답을 작성하세요.

### 명령어:
""",
        roles=("User", "Assistant", "System"),
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.GUANACO,
        sep="\n",
        sep2="\n\n",
    )
)

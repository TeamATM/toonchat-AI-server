import json
import logging
from pydantic import TypeAdapter

from app.message_queue.amqp import Amqp
from app.llm.utils import load_model
from app.llm.models import BaseLLM
from app.llm.conversations import get_conv_template
from app.data import PromptData

logger = logging.getLogger(__name__)


class InferenceTask:
    model: BaseLLM = None
    amqp: Amqp

    def __init__(self, amqp) -> None:
        self.model = load_model()
        self.amqp = amqp
        amqp.attach(self)

    def update(self, data):
        if isinstance(data, dict) and "id" in data:
            args = data.get("args", None)
            if isinstance(args, list) and 0 < len(args) < 3:
                answer, user_id = self.inference(
                    id=data.get("id"), data=args[0], stream=args[1] if len(args) == 2 else None
                )
                body = json.dumps(answer, ensure_ascii=False)
                self.publish(body, user_id)

    def inference(self, id: str, data: dict, stream=False):
        message: PromptData
        try:
            message = TypeAdapter(PromptData).validate_python(data)
        except Exception as e:
            logger.error("Failed to map message to dataclass. message: %s, error: %s", data, e)
            return

        conv = get_conv_template(self.model.promt_template)

        for m in message.get_chat_history_list()[:-1]:
            conv.append_message(conv.roles[0 if m.is_user() else 1], m.content)

        conv.append_message(conv.roles[2], message.get_persona())
        conv.append_message(conv.roles[0], message.get_chat_history_list()[-1].content)
        conv.append_message(conv.roles[1], None)

        streamer = self.model.generate(conv.get_prompt(), **message.get_generation_args())

        completion = []

        for token in streamer:
            completion.append(token)

        completion = ("".join(completion)).strip()
        return message.build_return_message(id, completion).to_dict(), message.get_user_id()

    def publish(self, data: str, routing_key: str):
        self.amqp.publish(routing_key, data)

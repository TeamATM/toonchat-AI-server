import json
import logging
from pydantic import TypeAdapter

from app.message_queue.amqp import Amqp
from app.message_queue.ampq_observer import AmqpObserver
from app.message_queue.data import PromptData
from app.llm.factory import llm_factory
from app.llm.config import llm_config

logger = logging.getLogger(__name__)


class InferenceTask(AmqpObserver):
    model = None
    amqp: Amqp

    def __init__(self, amqp: Amqp) -> None:
        self.model = llm_factory.create_llm(
            llm_config.prompt_template, llm_config.pretrained_model_name_or_path
        )
        self.amqp = amqp
        amqp.attach(self)

    def update(self, data):
        if isinstance(data, dict) and "id" in data:
            args = data.get("args", None)
            if isinstance(args, list) and 0 < len(args) < 3:
                answer, user_id = self.inference(id=data.get("id"), data=args[0])
                body = json.dumps(answer, ensure_ascii=False)
                self.publish(body, user_id)

    def inference(self, id: str, data: dict):
        message: PromptData
        try:
            message = TypeAdapter(PromptData).validate_python(data)
        except Exception as e:
            logger.error("Failed to map message to dataclass. message: %s, error: %s", data, e)
            return

        completion_result = self.model.generate(message, **message.get_generation_args())
        return message.build_return_message(id, completion_result).to_dict(), message.get_user_id()

    def publish(self, data: str, routing_key: str):
        self.amqp.publish(routing_key, data)

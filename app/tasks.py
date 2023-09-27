from celery.app.task import Context
from datetime import datetime
from celery import Task

from app.worker import app
from app.llm.utils import load_model
from app.llm.models import BaseLLM
from app.llm.conversations import get_conv_template
from app.data import PromptData, MessageToMq


class InferenceTask(Task):
    """
    Abstraction of Celery's Task class to support loading ML model.
    """

    model: BaseLLM = None

    def __init__(self) -> None:
        super().__init__()
        self.max_retries = 1
        # if not self.model:
        #     print("Load Model")
        #     self.model = load_model()

    def __call__(self, *args, **kwargs):
        """
        Load model on first call (i.e. first task processed)
        Avoids the need to load model on each task request
        """
        if not self.model:
            print("Start Model loading...")
            self.model = load_model()
            print("Finished to load model")

        return super().__call__(*args, **kwargs)


def publish(task: Task, data: MessageToMq, exchange: str, routing_key: str, **kwargs):
    request: Context = task.request
    with task.app.amqp.producer_pool.acquire(block=True) as producer:
        producer.publish(
            data.to_dict(),
            exchange=exchange,
            routing_key=routing_key,
            correlation_id=request.id,
            serializer=task.serializer,
            retry=True,
            retry_policy={
                "max_retries": 1,
                "interval_start": 0,
                "interval_step": 1,
                "interval_max": 1,
            },
            declare=None,
            delivery_mode=1,
        )


def build_message(messageId, content, user_id, character_id):
    message = MessageToMq(
        messageId=messageId,
        userId=user_id,
        characterId=character_id,
        createdAt=datetime.now(),
        content=content,
    )
    return message


@app.task(bind=True, base=InferenceTask, name="inference")
def inference(self: InferenceTask, data: dict, stream=False):
    request: Context = self.request
    message = PromptData(**data)

    exchange_name = "amq.topic"
    user_id = message.get_user_id()
    persona = message.get_persona()

    conv = get_conv_template(self.model.promt_template)

    for m in message.get_chat_history_list()[:-1]:
        conv.append_message(conv.roles[0 if m.is_user() else 1], m.content)

    conv.append_message(conv.roles[2], persona)
    conv.append_message(conv.roles[0], message.get_chat_history_list()[-1].content)
    conv.append_message(conv.roles[1], None)

    streamer = self.model.generate(conv.get_prompt(), **message.get_generation_args())

    completion = []

    for token in streamer:
        completion.append(token)
        if stream:
            publish(
                self,
                message.build_return_message(request.id, token),
                exchange_name,
                user_id,
            )

    completion = ("".join(completion)).strip()
    publish(self, message.build_return_message(request.id, completion), exchange_name, user_id)
    return completion

from celery.app.task import Context
from celery import Task, states
from time import time

from app.worker import app
from app.llm.utils import load_model
from app.llm.models import BaseLLM
from app.llm.conversations import get_conv_template


class InferenceTask(Task):
    """
    Abstraction of Celery's Task class to support loading ML model.
    """

    model: BaseLLM = None

    def __init__(self) -> None:
        super().__init__()
        self.max_retries = 1

    def __call__(self, *args, **kwargs):
        """
        Load model on first call (i.e. first task processed)
        Avoids the need to load model on each task request
        """
        if not self.model:
            print("Load Model")
            self.model = load_model()

        return super().__call__(*args, **kwargs)


def publish(task: Task, data: dict, exchange: str, routing_key: str, **kwargs):
    request: Context = task.request
    with task.app.amqp.producer_pool.acquire(block=True) as producer:
        producer.publish(
            data,
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


def get_data(messageId, status, content, data: dict):
    return {
        "messageId": messageId,
        "status": status,
        "content": content,
        "messageFrom": data.get("messageTo"),
        "messageTo": data.get("messageFrom"),
        "characterName": data.get("characterName"),
        "userId": data.get("userId", "userId is not in data!"),
        "createdAt": int(time() * 1000),
    }


@app.task(bind=True, base=InferenceTask, name="inference")
def inference(self: InferenceTask, data: dict, stream=False):
    request: Context = self.request

    conv = get_conv_template(self.model.promt_template)
    if data["history"]:
        for message in data["history"]:
            conv.append_message(
                conv.roles[0] if message["status"] == states.STARTED else conv.roles[1],
                message["content"],
            )

    conv.append_message(conv.roles[0], data["content"])
    conv.append_message(conv.roles[1], None)

    streamer = self.model.generate(
        conv.get_prompt(), bot=data.get("characterName", None), **data.get("parameters", {})
    )

    completion = []

    for token in streamer:
        completion.append(token)
        if stream:
            publish(
                self,
                get_data(request.id, "PROCESSING", token, data),
                "amq.topic",
                data["messageFrom"],
            )

    completion = ("".join(completion)).strip()
    publish(
        self,
        get_data(request.id, states.SUCCESS, completion, data),
        "amq.topic",
        data["messageFrom"],
    )
    return completion

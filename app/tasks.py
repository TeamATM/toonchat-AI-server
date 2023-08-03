from celery.app.task import Context
from celery import Task, states
from time import time

from app.worker import app
from app.llm.utils import load_model
from app.llm.models import BaseLLM
from app.llm.config import llm_config


class InferenceTask(Task):
    """
    Abstraction of Celery's Task class to support loading ML model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model: BaseLLM = None
        self.max_retries = 1

    def __call__(self, *args, **kwargs):
        """
        Load model on first call (i.e. first task processed)
        Avoids the need to load model on each task request
        """
        if not self.model:
            print("Load Model")
            # llm_config = LLMConfig(
            #     ModelType.LoRA,
            #     jbase_model_path=join("models", "beomi", "KoAlpaca-Polyglot-5.8B"),
            #     adapter_path=join("models", "checkpoint-2200"),
            #     prompt_fname="Remon",
            # )

            self.model = load_model(llm_config)

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


def get_data(messageId, status, content, chat_from, characterName):
    return {
        "messageId": messageId,
        "status": status,
        "content": content,
        "messageFrom": chat_from,
        "characterName": characterName,
        "createdAt": int(time() * 1000),
    }


@app.task(bind=True, base=InferenceTask, name="inference")
def inference(self: Task, data, stream=False):
    request: Context = self.request
    streamer = self.model.generate(
        history=data["history"], x=data["content"], bot=data["characterName"]
    )

    completion = []

    for token in streamer:
        completion.append(token)
        if stream:
            publish(
                self,
                get_data(request.id, "PROCESSING", token, data["messageTo"], data["characterName"]),
                "amq.topic",
                data["messageFrom"],
            )

    completion = "".join(completion)
    publish(
        self,
        get_data(request.id, states.SUCCESS, completion, data["messageTo"], data["characterName"]),
        "amq.topic",
        data["messageFrom"],
    )
    return completion

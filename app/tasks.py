from os.path import join
from celery.app.task import Context
from celery import Task, states

from app.worker import app
from app.llm.utils import load_model, load_lora
from app.llm.models import BaseLLM, LLMConfig
from app.llm.constants import ModelType


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
            llm_config = LLMConfig(
                ModelType.LoRA,
                join("models", "checkpoint-2200"),
                "Remon",
                join("models", "beomi", "KoAlpaca-Polyglot-5.8B"),
                stopping_words=["\n\n", "\nHuman:"],
            )
            self.model = load_model(llm_config)
            load_lora(self.model, llm_config.adapter_path)

        return super().__call__(*args, **kwargs)


def publish(task: Task, output: str, state: str, **kwargs):
    request: Context = task.request
    with task.app.amqp.producer_pool.acquire(block=True) as producer:
        producer.publish(
            {"taskId": request.id, "status": state, "result": (output,), **kwargs},
            exchange="response",
            routing_key=request.reply_to,
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


@app.task(bind=True, base=InferenceTask, name="inference")
def inference(self: Task, data):
    streamer = self.model.generate(history=data["history"], x=data["value"])

    completion = []

    for token in streamer:
        completion.append(token)
        publish(self, token, "PROCESSING")

    completion = "".join(completion)
    publish(self, completion, states.SUCCESS)
    return completion

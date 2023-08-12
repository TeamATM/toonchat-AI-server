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
        # if not self.model:
        #     print("Load Model")
        #     self.model = load_model()

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

    conv.append_message(
        conv.roles[2],
        "내 이름은 이영준이다. 1986년 6월 21일생인 33살 남자이다. 나는 유명그룹의 부회장이다. 나는 뛰어난 지능을 가지고 있다. 나는 매력적인 외모를 가지고 있다. 나는 카리스마가 있다. 나는 자기애가 강하다."
        if data.get("characterName") == "이영준"
        else "내 이름은 김미소이다. 나는 1990년 4월 5월생인 29살 여자이다. 나는 이영준 부회장의 개인 비서이다. 나는 뛰어난 지능을 가지고 있다. 나는 높은 의사소통 능력을 가지고 있다. 나는 일찍부터 사회생활에 뛰어들었다. 나는 퇴사를 고려중이다."
        if data.get("characterName") == "김미소"
        else "",
    )
    conv.append_message(conv.roles[0], data["content"])
    conv.append_message(conv.roles[1], None)

    streamer = self.model.generate(
        conv.get_prompt(), bot=data.get("characterName", None), **data.get("generationArgs", {})
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

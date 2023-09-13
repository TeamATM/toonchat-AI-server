from celery.app.task import Context
from celery import Task
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
            print("Start Model loading...")
            self.model = load_model()
            print("Finished to load model")

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


def build_message(messageId, content, user_id, character_id):
    return {
        "messageId": messageId,
        "userId": user_id,
        "characterId": character_id,
        "createdAt": int(time() * 1000),
        "content": content,
        "fromUser": False,
        # TODO: 아래 쓸모없는 정보들 제거
        "status": "SUCCESS",
        "messageFrom": character_id,
        "messageTo": "AAAAAAAAAAAA",
        "characterName": "이영준",
    }


@app.task(bind=True, base=InferenceTask, name="inference")
def inference(self: InferenceTask, data: dict, stream=False):
    request: Context = self.request

    exchange_name = "amq.topic"
    user_id = data.get("userId", "Anonymous")
    character_id = data.get("characterId", 0)
    persona = data.get("persona", "")

    if not persona:
        persona = (
            "내 이름은 이영준이다. 1986년 6월 21일생인 33살 남자이다. 나는 유명그룹의 부회장이다. 나는 뛰어난 지능을 가지고 있다. 나는 매력적인 외모를 가지고 있다. 나는 카리스마가 있다. 나는 자기애가 강하다."
            if character_id == 0
            else "내 이름은 김미소이다. 나는 1990년 4월 5월생인 29살 여자이다. 나는 이영준 부회장의 개인 비서이다. 나는 뛰어난 지능을 가지고 있다. 나는 높은 의사소통 능력을 가지고 있다. 나는 일찍부터 사회생활에 뛰어들었다. 나는 퇴사를 고려중이다."
            if character_id == 1
            else persona
        )
    elif isinstance(persona, list):
        persona = " ".join(persona)

    conv = get_conv_template(self.model.promt_template)
    for i, message in enumerate(data.get("history", [])):
        conv.append_message(
            conv.roles[0 if message.get("fromUser", i % 2 == 0) else 1], message["content"]
        )

    conv.append_message(conv.roles[2], persona)
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
                build_message(request.id, token, user_id, character_id),
                exchange_name,
                user_id,
            )

    completion = ("".join(completion)).strip()
    publish(
        self, build_message(request.id, completion, user_id, character_id), exchange_name, user_id
    )
    return completion

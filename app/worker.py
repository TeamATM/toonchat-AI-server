from os import chdir
from os.path import dirname

chdir(dirname(dirname(__file__)))

from celery import Celery
from app.config import config


app = Celery(
    "toonchat", broker=config.BROKER_URI, backend=config.BACKEND_URI, include=["app.tasks"]
)
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=86400,  # 하루
)
# app.conf.result_backend = 'backend.CustomRPCBackend'

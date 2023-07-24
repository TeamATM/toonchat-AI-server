from os import chdir, environ, name
from os.path import dirname

chdir(dirname(dirname(__file__)))

from celery import Celery
from app.config import config

if __name__ != "__main__":
    environ["FORKED_BY_MULTIPROCESSING"] = "1"
    if name != "nt":
        from billiard import context

        context._force_start_method("spawn")

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

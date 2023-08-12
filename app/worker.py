from os import chdir, environ, name
from os.path import dirname

chdir(dirname(dirname(__file__)))

from app.utils import get_profile
from env import get_celery_conf
from celery import Celery


if __name__ != "__main__":
    environ["FORKED_BY_MULTIPROCESSING"] = "1"
    if name != "nt":
        from billiard import context

        context._force_start_method("spawn")

app = Celery("toonchat", include=["app.tasks"])

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=86400,  # 하루
    **get_celery_conf(get_profile()),
)
# app.conf.result_backend = 'backend.CustomRPCBackend'

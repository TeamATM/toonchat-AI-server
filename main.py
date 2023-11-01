import logging

from app.message_queue.config import Config
from app.message_queue.amqp import Amqp
from app.tasks import InferenceTask

LOG_FORMAT = (
    "%(levelname) -10s %(asctime)s %(name) -30s %(funcName) " "-35s %(lineno) -5d: %(message)s"
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

amqp = Amqp(Config())
task = InferenceTask(amqp)
amqp.run()

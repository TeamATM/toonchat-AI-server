import logging
import json
import pika
import functools
from pika.exchange_type import ExchangeType
from pika.channel import Channel
from pika.spec import Basic, BasicProperties

from app.message_queue.config import Config

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


class Amqp(object):
    publish_exchange = "amq.topic"
    observers = {}

    def __init__(self, config: Config) -> None:
        self.should_reconnect = True
        self.was_consuming = False

        self._connection: pika.SelectConnection = None
        self._channel: Channel = None
        self._closing = False
        self._consumer_tag = None
        self._consuming = False
        self._url = config.broker_url
        self.consume_exchange = config.task_default_exchange
        self.consume_queue = config.task_default_queue
        self.consume_routing_key = config.task_default_routing_key
        self.consume_exchange_type = ExchangeType.direct

        self._prefetch_count = 1

    def attach(self, observer: object):
        logger.info("Observer %s attached", observer.__class__.__name__)
        self.observers[observer.__class__.__name__] = observer

    def detach(self, observer: object):
        logger.info("Observer %s detached", observer.__class__.__name__)
        self.observers.pop(observer.__class__.__name__, observer, None)

    def connect(self):
        logger.info("connection to %s", self._url)
        return pika.SelectConnection(
            parameters=pika.URLParameters(self._url),
            on_open_callback=self.on_connection_open,
            on_open_error_callback=self.on_connection_open_error,
            on_close_callback=self.on_connection_closed,
        )

    def on_connection_open(self, _unused_connection: pika.SelectConnection):
        logger.info("Connection opened")
        self.open_channel()

    def open_channel(self):
        logger.info("Creating a new channel")
        self._connection.channel(on_open_callback=self.on_channel_open)

    def on_channel_open(self, channel):
        logger.info("Channel opened")
        self._channel = channel
        self.add_on_channel_close_callback()
        self.setup_exchange(self.consume_exchange)

    def add_on_channel_close_callback(self):
        logger.info("Adding channel close callback")
        self._channel.add_on_close_callback(self.on_channel_closed)

    def on_channel_closed(self, channel, reason):
        logger.warning("Channel %i was closed: %s", channel, reason)
        self.close_connection()

    def on_connection_open_error(self, _unused_connection: pika.SelectConnection, err: Exception):
        logger.error("Connection open failed: %s", err)
        self.reconnect()

    def on_connection_closed(self, _unused_connection: pika.SelectConnection, reason: Exception):
        self._channel = None
        if self._closing:
            self._connection.ioloop.stop()
        else:
            logger.warning("Connection closed, reconnect necessary: %s", reason)
            self.reconnect()

    def close_connection(self):
        self._consuming = False
        if self._connection.is_closing or self._connection.is_closed:
            logger.info("Connection is closing or already closed")
        else:
            logger.info("Closing connection")
            self._connection.close()

    def reconnect(self):
        self.should_reconnect = True
        self.stop()

    def stop(self):
        if not self._closing:
            self._closing = True
            logger.info("Stopping")
            if self._consuming:
                self.stop_consuming()
                self._connection.ioloop.start()
            else:
                self._connection.ioloop.stop()
            logger.info("Stopped")

    def setup_exchange(self, exchange_name):
        logger.info("Declaring exchange: %s", exchange_name)
        # Note: using functools.partial is not required, it is demonstrating
        # how arbitrary data can be passed to the callback when it is called
        cb = functools.partial(self.on_exchange_declareok, userdata=exchange_name)
        self._channel.exchange_declare(
            exchange=exchange_name,
            exchange_type=self.consume_exchange_type,
            callback=cb,
            durable=True,
        )

    def on_exchange_declareok(self, _unused_frame, userdata):
        logger.info("Exchange declared: %s", userdata)
        self.setup_queue(self.consume_queue)

    def setup_queue(self, queue_name):
        logger.info("Declaring queue %s", queue_name)
        cb = functools.partial(self.on_queue_declareok, userdata=queue_name)
        self._channel.queue_declare(queue=queue_name, callback=cb, durable=True)

    def on_queue_declareok(self, _unused_frame, userdata):
        queue_name = userdata
        logger.info(
            "Binding %s to %s with %s", self.consume_exchange, queue_name, self.consume_routing_key
        )
        cb = functools.partial(self.on_bindok, userdata=queue_name)
        self._channel.queue_bind(
            queue_name, self.consume_exchange, routing_key=self.consume_routing_key, callback=cb
        )

    def on_bindok(self, _unused_frame, userdata):
        logger.info("Queue bound: %s", userdata)
        self.set_qos()

    def set_qos(self):
        self._channel.basic_qos(prefetch_count=self._prefetch_count, callback=self.on_basic_qos_ok)

    def on_basic_qos_ok(self, _unused_frame):
        logger.info("QOS set to: %d", self._prefetch_count)
        self.start_consuming()

    def start_consuming(self):
        logger.info("Issuing consumer related RPC commands")
        self.add_on_cancel_callback()
        self._consumer_tag = self._channel.basic_consume(self.consume_queue, self.on_message)
        self.was_consuming = True
        self._consuming = True

    def add_on_cancel_callback(self):
        logger.info("Adding consumer cancellation callback")
        self._channel.add_on_cancel_callback(self.on_consumer_cancelled)

    def on_consumer_cancelled(self, method_frame):
        logger.info("Consumer was cancelled remotely, shutting down: %r", method_frame)
        if self._channel:
            self._channel.close()

    def on_message(
        self,
        _unused_channel: pika.SelectConnection,
        basic_deliver: Basic.Deliver,
        properties: BasicProperties,
        body: bytes,
    ):
        try:
            message = json.loads(str(body, encoding="utf-8"))
        except Exception as e:
            logger.error("Not a valid json format: %s", body)
            self.reject_message(basic_deliver.delivery_tag, e, False)
            return

        logger.info(
            "Received message # %s from %s: %s",
            basic_deliver.delivery_tag,
            properties.app_id,
            message,
        )

        try:
            for observer in self.observers.values():
                observer.update(message)
            self.acknowledge_message(basic_deliver.delivery_tag)
        except Exception as e:
            self.reject_message(basic_deliver.delivery_tag, e)

    def acknowledge_message(self, delivery_tag):
        logger.info("Acknowledging message %s", delivery_tag)
        self._channel.basic_ack(delivery_tag)

    def reject_message(self, delivery_tag, exception: Exception, requeue=False):
        logger.info("Rejecting message %s by: ", delivery_tag, exception)
        self._channel.basic_nack(delivery_tag, requeue=requeue)

    def stop_consuming(self):
        if self._channel:
            logger.info("Sending a Basic.Cancel RPC command to RabbitMQ")
            cb = functools.partial(self.on_cancelok, userdata=self._consumer_tag)
            self._channel.basic_cancel(self._consumer_tag, cb)

    def on_cancelok(self, _unused_frame, userdata):
        self._consuming = False
        logger.info("RabbitMQ acknowledged the cancellation of the consumer: %s", userdata)
        self.close_channel()

    def close_channel(self):
        logger.info("Closing the channel")
        self._channel.close()

    def run(self):
        self._connection = self.connect()
        self._connection.ioloop.start()

    def publish(self, routing_key, body):
        logger.info("Publishing message to user: %s, message: %s", routing_key, body)
        try:
            property = BasicProperties(content_type="application/json", content_encoding="utf-8")
            self._channel.basic_publish(self.publish_exchange, routing_key, body, property)
        except Exception as e:
            logger.error(
                "Failed to publish message. user: %s, message: %s, error: %s", routing_key, body, e
            )

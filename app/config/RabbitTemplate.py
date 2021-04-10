import os
import json
import pika
from typing import Optional

from app.config.constants.RabbitConstants import RabbitConstants


class RabbitTemplate:
    RABBIT_USER: Optional[str]
    RABBIT_PW: Optional[str]
    RABBIT_HOST: Optional[str]
    RABBIT_PORT: Optional[int]

    def __init__(self):
        self.load_env()
        credentials = pika.PlainCredentials(self.RABBIT_USER, self.RABBIT_PW)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=self.RABBIT_HOST, port=int(self.RABBIT_PORT), credentials=credentials))
        self.channel = connection.channel()

        # Declare RPC
        self.channel.queue_declare(queue=RabbitConstants.RPC_REQUEST_QUEUE)
        self.channel.exchange_declare(RabbitConstants.RPC_DIRECT_EXCHANGE, exchange_type='direct', durable=True)
        self.channel.queue_bind(queue=RabbitConstants.RPC_REQUEST_QUEUE,
                                exchange=RabbitConstants.RPC_DIRECT_EXCHANGE,
                                routing_key=RabbitConstants.RPC_REQ_ROUTING_KEY)

    # .env vars
    def load_env(self):
        self.RABBIT_USER = os.getenv('RABBIT_USER')
        self.RABBIT_PW = os.getenv('RABBIT_PW')
        self.RABBIT_HOST = os.getenv('RABBIT_HOST')
        self.RABBIT_PORT = os.getenv('RABBIT_PORT')

    def start_consuming(self, method):
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=RabbitConstants.RPC_REQUEST_QUEUE,
                                   on_message_callback=method)

    def rpc_reply(self, ch, method, props, body):
        ch.basic_publish(exchange='',
                         routing_key=props.reply_to,
                         properties=pika.BasicProperties(correlation_id=props.correlation_id),
                         body=body)
        ch.basic_ack(delivery_tag=method.delivery_tag)

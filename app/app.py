import logging
from pathlib import Path
from dotenv import load_dotenv
import sys

workdir = '/home/csantos/src/aidl/video_processor'
sys.path.insert(0, workdir)

from app.config.RabbitTemplate import RabbitTemplate
from app.controller.Controller import Controller

def load_env():
    env_path = Path(workdir + '/env') / '.env'
    load_dotenv(dotenv_path=env_path)


if __name__ == '__main__':
    # Load environment vars
    load_env()

    # Create RPC Controller and start consumer
    rabbit_template = RabbitTemplate()
    controller = Controller(rabbit_template)
    controller.start()
    print(" [X] Start Consuming: ")
    rabbit_template.channel.start_consuming()

import logging
from pathlib import Path
from dotenv import load_dotenv


from app.controller.Controller import Controller
from app.config.RabbitTemplate import RabbitTemplate


def load_env():
    env_path = Path('env') / '.env'
    load_dotenv(dotenv_path=env_path)


if __name__ == '__main__':
    # Load environment vars
    load_env()

    # Create RPC Controller and start consumer
    controller = Controller()
    logging.info(" [X] Start Consuming: ")
    controller.start()


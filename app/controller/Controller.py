from app.GesturesAnalyzer.GesturesAnalyzer import GesturesAnalyzer
from app.config.RabbitTemplate import RabbitTemplate


class Controller:

    def __init__(self):
        self.rabbit_template = RabbitTemplate()
        self.gestures_analyzer = GesturesAnalyzer()

    def start(self):
        self.rabbit_template.start_consuming(self.consume_video)

    def consume_video(self, ch, method, props, body):
        response = self.gestures_analyzer.process_video(video=body)
        self.rabbit_template.rpc_reply(ch, method, props, response)

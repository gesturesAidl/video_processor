import datetime

from app.GesturesAnalyzer.GesturesAnalyzer import GesturesAnalyzer

class Controller:

    def __init__(self, rabbit_template):
        self.rabbit_template = rabbit_template
        self.gestures_analyzer = GesturesAnalyzer()

    def start(self):
        self.rabbit_template.start_consuming(self.consume_video)

    def consume_video(self, ch, method, props, body):
        print("Video received at: " + str(datetime.datetime.utcnow()))
        response = self.gestures_analyzer.process_video(video=body)
        self.rabbit_template.rpc_reply(ch, method, props, response)

import datetime
import json
import os

from app.GesturesAnalyzer.GesturesAnalyzer import GesturesAnalyzer


class Controller:

    def __init__(self, rabbit_template):
        self.rabbit_template = rabbit_template
        self.gestures_analyzer = GesturesAnalyzer()

    def init(self):
        self.rabbit_template.start_consuming(self.consume_video)

    def consume_video(self, ch, method, props, body):
        print("Video received at: " + str(datetime.datetime.utcnow()))
        frames = json.loads(body)

        path = self.gestures_analyzer.save_video(frames)
        if os.path.exists(path):
            label = self.gestures_analyzer.process_video(path)
            json_response = {'label': label}

        self.rabbit_template.rpc_reply(ch, method, props, json.dumps(json_response))

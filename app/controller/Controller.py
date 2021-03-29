import datetime
import json

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
        # TODO: replace .save_video(frames) method by process_video(frames) when its code has been implemented.
        #  With save_video code, we can check that videos are sent and saved correctly
        response = self.gestures_analyzer.save_video(frames)
        # response = self.gestures_analyzer.process_video(video=frames)
        self.rabbit_template.rpc_reply(ch, method, props, response)

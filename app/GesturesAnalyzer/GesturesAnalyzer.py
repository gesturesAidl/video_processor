import threading
import numpy as np
import cv2
import os

from app.GesturesAnalyzer.FeatureExtractor import FeatureExtractor
from app.GesturesAnalyzer.OpticalFlowExtractor import OpticalFlowExtractor
from app.domain.Gestures import Gestures


class GesturesAnalyzer:

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.optical_flow_extractor = OpticalFlowExtractor()
        self.count = 0
        self.last_clip = None
        self.frameSize = (170, 100)
        self.fps = 12
        self.video_writer = cv2.VideoWriter_fourcc(*'XVID')

    def process_video(self, video):
        try:
            threads = []
            feature_extraction = threading.Thread(target=self.feature_extractor.extract_features, args=[video])
            threads.append(feature_extraction)
            feature_extraction.start()

            optical_flow_calc = threading.Thread(target=self.optical_flow_extractor.extract_optical_flow, args=[video])
            threads.append(optical_flow_calc)
            optical_flow_calc.start()

            # Blocking main thread until both processes have been performed.
            features = feature_extraction.join()
            optical_flow = optical_flow_calc.join()

        except Exception as e:
            print(str(e))

        """
        ### TODO:
        When both methods have finished, use results in {@features} and {@optical_flow} to pass through model to get the 
        most likely class of the video.
        Set results as a Gestures object
        """

        return Gestures()

    def save_video(self, frames):
        self.count = self.count+1
        video_name = str(self.count) + '.avi'
        path = os.getenv('VIDEOS_OUT_PATH')

        if self.last_clip:
            video = self.last_clip + frames
            video_out = cv2.VideoWriter(path+video_name, self.video_writer, self.fps, self.frameSize)
            for frame in video:
                video_out.write(np.array(frame, dtype='uint8'))
            video_out.release()

        self.last_clip = frames
        return Gestures()
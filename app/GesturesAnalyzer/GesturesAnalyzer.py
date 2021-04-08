import threading
import numpy as np
import cv2
import os
import time

from app.config import config
from app.GesturesAnalyzer.FeatureExtractor import FeatureExtractor
from app.GesturesAnalyzer.OpticalFlowExtractor import OpticalFlowExtractor
from app.GesturesAnalyzer.PredictGesture import PredictGesture
from app.domain.Gestures import Gestures


class GesturesAnalyzer:

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.optical_flow_extractor = OpticalFlowExtractor()
        self.predict_gesture = PredictGesture()
        self.count = 0
        self.last_clip = None
        self.frameSize = (170, 100)
        self.fps = 12
        self.video_writer = cv2.VideoWriter_fourcc(*'XVID')

    def process_video(self, path):
        start = time.time()
        try:
            features = self.feature_extractor.extract_features(path, 0)
            optical_flow_path = self.optical_flow_extractor.extract_optical_flow(path,1)
            optical_flow_feat = self.feature_extractor.extract_features(optical_flow_path, 1)

        except Exception as e:
            print(str(e))

        # When both methods have finished, use results in {@features} and {@optical_flow} to pass through
        # model to get the most likely class of the video.
        
        pred_gesture = self.predict_gesture.prediction(features, optical_flow_feat)        
        # Set results as a Gestures object
        gesture = Gestures()
        gesture.set_label(pred_gesture)

        end = time.time()
        print("Process video:" + str(end-start))
        return gesture

    def save_video(self, frames):
        start = time.time()
        self.count = self.count+1
        video_name = str(self.count) + '.avi'
        path = os.getenv('VIDEOS_OUT_PATH')

        if self.last_clip:
            video = self.last_clip + frames
            video_out = cv2.VideoWriter(path+'/'+video_name, self.video_writer, self.fps, self.frameSize)
            for frame in video:
                video_out.write(np.array(frame, dtype='uint8'))
            video_out.release()

        self.last_clip = frames

        end = time.time()
        print("Save video:" + str(end - start))
        #TODO: return both as workaround, until better way to handle responses (See Controller)
        return Gestures(), path+'/'+video_name

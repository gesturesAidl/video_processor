import threading
import numpy as np
import cv2
import os
import time
import json

from app.GesturesAnalyzer.FeatureExtractor import FeatureExtractor
from app.GesturesAnalyzer.OpticalFlowExtractor import OpticalFlowExtractor
from app.GesturesAnalyzer.PredictGesture import PredictGesture
from app.domain.Gestures import Gestures

global features

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
        self.labels = ["no_gesture", "doing_other_things", "stop_sign", "thumb_up", "sliding_two_fingers_down", 
            "sliding_two_fingers_up", "swiping_right", "swiping_left", "turning_hand_clockwise"]

    def process_video(self, path):
        start = time.time()
        features = []
        optical_flow_feat = []
        try:
            # threads = []
            # feature_extraction = threading.Thread(target=self.feature_extractor.extract_features, args=[path, 0])
            # threads.append(feature_extraction)
            # feature_extraction.start()

            # optical_flow_calc = threading.Thread(target=self.optical_flow_extractor.extract_optical_flow, args=[path, 1])
            # threads.append(optical_flow_calc)
            # optical_flow_calc.start()

            features = self.feature_extractor.extract_features(path, 0)
            optical_flow_path = self.optical_flow_extractor.extract_optical_flow(path,1)
            optical_flow_feat = self.feature_extractor.extract_features(optical_flow_path, 1)
            # # Blocking main thread until both processes have been performed.
            # feature_extraction.join()
            # optical_flow_calc.join()
            # for t in threads:
            #     print(t)
            #     features[t._id] = t.video_feat
        except Exception as e:
            print(str(e))

        # When both methods have finished, use results in {@features} and {@optical_flow} to pass through
        # model to get the most likely class of the video.

        pred_gesture = self.predict_gesture.prediction(features, optical_flow_feat)

        # Set results as a Gestures object
        #gesture = Gestures()
        #gesture.set_label(pred_gesture)
        
        gesture = json.dumps({'label': self.labels[pred_gesture]})

        end = time.time()
        print("Process video:" + str(end-start))
        return gesture

    def save_video(self, frames):
        self.count = self.count+1 % 10
        video_name = str(self.count) + '.avi'
        path = os.getenv('VIDEOS_OUT_PATH')

        video_out = cv2.VideoWriter(path+'/'+video_name, self.video_writer, self.fps, self.frameSize)
        for frame in frames:
            video_out.write(np.array(frame, dtype='uint8'))
        video_out.release()
        return path+'/'+video_name

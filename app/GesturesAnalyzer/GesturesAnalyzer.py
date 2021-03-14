import threading

from app.GesturesAnalyzer.FeatureExtractor import FeatureExtractor
from app.GesturesAnalyzer.OpticalFlowExtractor import OpticalFlowExtractor
from app.domain.Gestures import Gestures


class GesturesAnalyzer:

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.optical_flow_extractor = OpticalFlowExtractor()

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


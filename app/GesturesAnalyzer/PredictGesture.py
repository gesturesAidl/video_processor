import torch
import os
import numpy as np
import time

from app.GesturesAnalyzer.Classifier2stream import ClassifierTwoStream


class PredictGesture:

    def __init__(self):

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Load model 
        self.trained_model = ClassifierTwoStream().to(self.device)

        # Load state_dict
        model_state_file = os.path.join(os.getenv("MODELS_DIR"), 'model_state_dict.pt')
        checkpoint = torch.load(model_state_file, map_location=self.device)
        self.trained_model.load_state_dict(checkpoint['model_state_dict'])
        self.trained_model.eval()

    def prediction(self, rgb_feat, flow_feat):
        start = time.time()
        #return True
        # Predict class from features
        with torch.no_grad():
            #TODO: Here code crashes. 
            # rgb_feat and flow_feat are of type NDArray - mxnet own version of numpy array
            # Find out how to fix...or convert to tensor
            rgb_feat = torch.tensor(rgb_feat).view(-1)
            rgb_feat = rgb_feat.to(self.device)
            flow_feat = torch.tensor(flow_feat).view(-1)
            flow_feat = flow_feat.to(self.device)
            output = self.trained_model(rgb_feat, flow_feat)
            pred_gesture = output.argmax(-1)

        end = time.time()
        print("Prediction:" + str(end - start))
        return pred_gesture

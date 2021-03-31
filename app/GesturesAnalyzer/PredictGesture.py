import torch
import os

from app.GesturesAnalyzer.Classifier2stream import ClassifierTwoStream


class PredictGesture:

    def __init__(self):

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Load model 
        self.trained_model = ClassifierTwoStream().to(self.device)

        # Load state_dict
        model_state_file = os.getenv("MODELS_DIR") + '/model_state_dict.pt'
        checkpoint = torch.load(model_state_file, map_location=self.device)
        self.trained_model.load_state_dict(checkpoint['model_state_dict'])
        self.trained_model.eval()

    def prediction(self, rgb_feat, flow_feat):
        
        # Predict class from features
        with torch.no_grad():
            rgb_feat = rgb_feat.to(self.device)
            flow_feat = flow_feat.to(self.device)
            output = self.trained_model(rgb_feat, flow_feat)
            pred_class = output.argmax(-1)

        return pred_gesture

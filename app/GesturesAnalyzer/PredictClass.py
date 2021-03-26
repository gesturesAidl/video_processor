import torch

from app.GesturesAnalyzer.Classifier2stream import ClassifierTwoStream

class PredictClass:

    def __init__(self):

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Load model 
        trained_model = ClassifierTwoStream().to(device)

        # Load state_dict
        model_state_file = 'model_state_dict.pt'
        checkpoint = torch.load(model_state_file, map_location=device)
        trained_model.load_state_dict(checkpoint['model_state_dict'])
        trained_model.eval()

    def prediction(self, rgb_feat, flow_feat):
        
        # Predict class from features
        with torch.no_grad():
            rgb_feat = rgb_feat.to(device)
            flow_feat = flow_feat.to(device)
            output = trained_model(rgb_feat, flow_feat)
            pred_class = output.argmax(-1)

        return pred_class

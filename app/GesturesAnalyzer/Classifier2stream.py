import torch.nn as nn

class ClassifierTwoStream(nn.Module):           # Makes use of "sum AFTER" method for features fusion

    def __init__(self, n_inputs=2048, h_rgb=1024, h_flow=1024, dropout=0.5, n_classes=9):
        super().__init__()
        self.fc_rgb = nn.Sequential(
                                 nn.Linear(n_inputs, h_rgb),       # input: Tensor 1x2048 = 2048 elements
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(h_rgb, n_classes),
                                )
        self.fc_flow = nn.Sequential(
                                 nn.Linear(n_inputs, h_flow),       # input: Tensor 1x2048 = 2048 elements
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(h_flow, n_classes),
                                )

    def forward(self, rgb, flow):
        rgb = self.fc_rgb(rgb)
        flow = self.fc_flow(flow)
        x = rgb + flow
        return x
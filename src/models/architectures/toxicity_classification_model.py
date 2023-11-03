import torch.nn as nn
import torch.nn.functional as F


class ToxicityClassificationModel(nn.Module):
    def __init__(self, input_dim):
        super(ToxicityClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(input_dim, 300)
        self.model = nn.Sequential(
            nn.Linear(300, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 250),
            nn.ReLU(True),
            nn.Linear(250, 50),
            nn.ReLU(True),
            nn.Linear(50, 1)
        )

    def forward(self, text):
        text = self.embedding(text)
        return F.sigmoid(self.model(text))

from torch import nn

class Classifier(nn.Module):
    def __init__(self, num_features=512, num_classes=1):
        super().__init__()

        self.linear = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.linear(x)
        return x

    
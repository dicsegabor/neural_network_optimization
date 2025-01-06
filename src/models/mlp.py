import torch.nn as nn


class MLPRegression(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_layers=(128, 64, 32),
        activation=nn.ReLU,
        dropout_rate=0.0,
        batch_norm=False,
    ):
        super(MLPRegression, self).__init__()
        layers = []
        in_features = input_size

        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(activation())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = out_features

        layers.append(nn.Linear(in_features, 1))  # Output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

"""Neural network architectures for RFID localization."""

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, output_size),
        )

    def forward(self, x):
        return self.linear_stack(x)


class ReLUModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_units=128):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_size),
        )

    def forward(self, x):
        return self.linear_stack(x)


class LeakyReLUModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_units=128):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_units // 2, output_size),
        )

    def forward(self, x):
        return self.linear_stack(x)


class LeakyReLUModel2(nn.Module):
    """Single hidden layer LeakyReLU model."""
    def __init__(self, input_size, output_size, hidden_units=64):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_units, output_size),
        )

    def forward(self, x):
        return self.linear_stack(x)


class LeakyReLUModel4(nn.Module):
    """Four-layer LeakyReLU model."""
    def __init__(self, input_size, output_size, hidden_units=64):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_units, output_size),
        )

    def forward(self, x):
        return self.linear_stack(x)


class LeakyReLUModelDropout(nn.Module):
    def __init__(self, input_size, output_size, hidden_units=256):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_units // 2, output_size),
        )

    def forward(self, x):
        return self.linear_stack(x)


class ReLUModelDropout(nn.Module):
    def __init__(self, input_size, output_size, hidden_units=256):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_units, output_size),
        )

    def forward(self, x):
        return self.linear_stack(x)


class TanhModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_units=64):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, output_size),
        )

    def forward(self, x):
        return self.linear_stack(x)


class SigmoidModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_units=64):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.Sigmoid(),
            nn.Linear(hidden_units, hidden_units),
            nn.Sigmoid(),
            nn.Linear(hidden_units, output_size),
        )

    def forward(self, x):
        return self.linear_stack(x)


class CNN1DModel(nn.Module):
    def __init__(self, input_channels=8, output_size=3):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.conv_stack(x)
        x = x.view(x.size(0), -1)
        return self.fc(self.dropout(x))


class EnhancedRNN(nn.Module):
    def __init__(self, input_size=16, hidden_size=128, num_layers=3, output_size=3, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=False,
            dropout=0 if num_layers == 1 else dropout / 2,
        )
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.act = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.act(self.fc1(self.dropout(out)))
        return self.fc2(out)


class FlexibleMLP(nn.Module):
    """MLP with a configurable list of hidden layer sizes and activation function."""
    def __init__(self, input_size, output_size, hidden_units=(128, 128), act_func=None):
        super().__init__()
        if act_func is None:
            act_func = nn.LeakyReLU(0.1)
        layers = []
        in_features = input_size
        for hidden in hidden_units:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(act_func)
            in_features = hidden
        layers.append(nn.Linear(in_features, output_size))
        self.linear_stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear_stack(x)


# Registry — maps name → class for use from the CLI
MODEL_REGISTRY = {
    "simple":           SimpleModel,
    "relu":             ReLUModel,
    "leaky_relu":       LeakyReLUModel,
    "leaky_relu2":      LeakyReLUModel2,
    "leaky_relu4":      LeakyReLUModel4,
    "leaky_relu_drop":  LeakyReLUModelDropout,
    "relu_drop":        ReLUModelDropout,
    "tanh":             TanhModel,
    "sigmoid":          SigmoidModel,
    "cnn":              CNN1DModel,
    "rnn":              EnhancedRNN,
    "mlp":              FlexibleMLP,
}

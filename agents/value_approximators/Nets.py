import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class Params:
    """
    This class allows to define the input and output of each neural network
    """
    input_dim: int
    hidden_dim: int
    n_hidden: int
    dropout: float
    seed: int
    output_dim: int


class Layers:
    def __init__(self, params: Params):
        self.params = params
        torch.manual_seed(self.params.seed)

    def dropout(self):
        return nn.Dropout(p=self.params.dropout)

    def lstm(self):
        return nn.LSTM(input_size=self.params.input_dim,
                               hidden_size=self.params.hidden_dim,
                               batch_first=True)

    def linear(self):
        return nn.Linear(in_features=self.params.input_dim, out_features=self.params.hidden_dim)

    def relu(self):
        return nn.ReLU()

    def set_linear(self, stack_layers: list):
        stack_layers.append(self.linear())
        stack_layers.append(self.relu())
        stack_layers.append(self.dropout())
        return stack_layers

    def set_lstm(self, stack_layers: list):
        stack_layers.append(self.lstm())
        stack_layers.append(self.dropout())
        return stack_layers

    def set(self, _type: str = None, stack_layers: list = None):
        if _type == 'DNN':
            stack_layers = self.set_linear(stack_layers)
        elif _type == 'LSTM':
            stack_layers = self.set_lstm(stack_layers)
        return stack_layers

    def get(self, _type: str = None):
        stack_layers = []
        stack_layers = self.set(_type, stack_layers)
        for i in range(self.params.n_hidden):
            self.params.input_dim = self.params.hidden_dim
            stack_layers = self.set(_type, stack_layers)
        return stack_layers[:-1] #Dropout layer on the outputs of each layer except the last layer


class DNN(nn.Module):
    def __init__(self, params: Params):
        super(DNN, self).__init__()
        self.params = params
        self.stack_layers = Layers(params).get('DNN')
        self.linear = nn.Linear(self.params.hidden_dim, self.params.output_dim)
        self.compute_set()

    def compute_set(self):
        for i, layer in enumerate(self.stack_layers):
            name = f'Layer_{str(i + 1).zfill(3)}'
            setattr(self, name, layer)

    def forward(self, x):
        for i, layer in enumerate(self.stack_layers):
            x = getattr(self, f'Layer_{str(i + 1).zfill(3)}')(x)
        x = self.linear(x[0])
        return x


class LSTM(nn.Module):
    def __init__(self, params: Params):
        super(LSTM, self).__init__()
        self.params = params
        self.stack_layers = Layers(params).get('LSTM')
        self.linear = nn.Linear(self.params.hidden_dim, self.params.output_dim)
        self.compute_set()

    def compute_set(self):
        for i, layer in enumerate(self.stack_layers):
            name = f'Layer_{str(i + 1).zfill(3)}'
            setattr(self, name, layer)

    def forward(self, x):
        for i, layer in enumerate(self.stack_layers):
            if isinstance(layer, nn.LSTM):
                x, (hn, cn) = getattr(self, f'Layer_{str(i + 1).zfill(3)}')(x)
            else:
                x = getattr(self, f'Layer_{str(i + 1).zfill(3)}')(x)
        x = self.linear(x[:, -1, :])
        return x
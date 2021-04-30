import torch
from torch import nn


class DrivingAlertLSTM(nn.Module):

    def __init__(self, hidden_size=10, bidirectional=False, num_layers=2, features_num=27):
        super(DrivingAlertLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=features_num,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5,
        )
        self.num_directional = 2 if bidirectional else 1
        self.output_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size * self.num_directional, 1),
            nn.Sigmoid()
        )
        self.epoch = 0

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        hidden = torch.cat([hidden[-i] for i in range(1, self.num_directional + 1)], dim=1)
        out = self.output_layer(hidden)
        return out.squeeze(1)

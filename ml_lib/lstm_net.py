import numpy as np

import torch
import torch.nn as nn

class LSTM_Net(nn.Module):
    """
    Pytorch LSTM Network
    """
    def __init__(self,
                 fc,
                 input_dim,
                 output_dim,
                 lstm_hidden_dim,
                 lstm_n_layers,
                 drop_prob=0.5,
                 out_actv=None):
        super(LSTM_Net, self).__init__()
        
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.dropout = nn.Dropout(drop_prob)
        if n_layers == 1: drop_prob = 0
        self.actv = nn.Relu()
        self.out_actv = out_actv

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        prev_out = hidden_dim
        self.fc = []
        for i,neurons in enumerate(fc):
            self.fc.append(nn.Linear(prev_out, neurons))
            prev_out = neurons
        self.out = nn.Linear(prev_out, output_size)

    def forward(self, x):
        # x = x.long()
        batch_size = x.size(0)
        lstm_out, self.hidden = self.lstm(x, self.init_hidden(batch_size))
        lstm_out = lstm_out[:, -1, :]

        dense = lstm_out
        for l in self.fc:
            dense = l(dense)
            dense = self.actv(dense)
            dense = self.dropout(dense)
        out = self.out(dense)
        if self.out_actv: out = self.out_atcv(out)
        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size,
                             self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size,
                                 self.hidden_dim).zero_())
        return hidden

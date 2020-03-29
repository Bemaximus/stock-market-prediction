import torch
import torch.nn as nn

class LSTM_Net(nn.Module):
    """
    Pytorch LSTM Network

    Attributes:
        n_layers: Number of layers in the LSTM. Used to initialize hidden state
        hidden_dim. Size of hidden dimension of LSTM. Used to initialize hidden
        state
        dropout: Dropout operation
        actv: Activation function. Defaults to ReLU
        out_actv: Activation function for the output. Assumes None unless passed
        lstm: LSTM layer or layers of the network
        fc: List of intermediate linear layers
        out: Output layer
    """
    def __init__(self,
                 fc,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 n_layers,
                 drop_prob=0.5,
                 out_actv=None):
        """
        Constructor.

        Parameters:
            fc: Iterable containing the number of neurons for each layer
            input_dim: Input dimensions
            output_dim: Output dimensions
            hidden_dim: LSTM hidden dimensions
            n_layers: Number of LSTM layers
            drop_prob: Probability to use for dropout layers. Defaults to 0.5
            out_actv: Pytroch activation function to use on final output. Is not
            called if None
        """
        super(LSTM_Net, self).__init__()
        
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.dropout = nn.Dropout(drop_prob)
        if n_layers == 1: drop_prob = 0
        self.actv = nn.Relu()
        self.out_actv = out_actv

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        prev_out = hidden_dim
        self.fc = []
        for i,neurons in enumerate(fc):
            self.fc.append(nn.Linear(prev_out, neurons))
            prev_out = neurons
        self.out = nn.Linear(prev_out, output_size)

    def forward(self, x):
        """
        Forward pass of the network

        Parameters:
            x: Input of size (batch, seq_length, input_size)

        Returns:
            Tensor: Output of size (batch, output_size)
        """
        x = x.long()
        batch_size = x.size(0)
        lstm_out, self.hidden = self.lstm(x, self.init_hidden(batch_size))
        # Save only the final hidden state for each sequence
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
        """
        Initializes an empty hidden state

        Parameters:
            batch_size: Batch size this hidden state will be used for

        Returns:
            Tuple: Hidden state composed of two tensors of size (n_layers,
            batch_size, hidden_dim)
        """
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size,
                             self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size,
                                 self.hidden_dim).zero_())
        return hidden

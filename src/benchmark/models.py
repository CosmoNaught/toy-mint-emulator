import torch.nn as nn

# Model Definition
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super(FFNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        self.layer3 = nn.Linear(output_size, output_size)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.softplus(x)
    
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, dropout=dropout_prob)  # Dropout between RNN layers
        self.fc = nn.Linear(hidden_size, output_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)  # Dropout before the FC layer
        self.softplus = nn.Softplus()

    def forward(self, x):
        x, _ = self.gru(x.view(len(x), 1, -1))
        x = self.ln(x)
        x = self.dropout(x)
        x = self.fc(x.view(len(x), -1))
        x = self.softplus(x)
        return x

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, dropout=dropout_prob)  # Dropout between LSTM layers
        self.fc = nn.Linear(hidden_size, output_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)  # Dropout before the FC layer
        self.softplus = nn.Softplus()

    def forward(self, x):
        x, _ = self.lstm(x.view(len(x), 1, -1))
        x = self.ln(x)
        x = self.dropout(x)
        x = self.fc(x.view(len(x), -1))
        x = self.softplus(x)
        return x

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, bidirectional=True, dropout=dropout_prob)  # Dropout between RNN layers
        self.fc = nn.Linear(2 * hidden_size, output_size)
        self.ln = nn.LayerNorm(2 * hidden_size)
        self.dropout = nn.Dropout(dropout_prob)  # Dropout before the FC layer
        self.softplus = nn.Softplus()

    def forward(self, x):
        x, _ = self.rnn(x.view(len(x), 1, -1))
        x = self.ln(x)
        x = self.dropout(x)
        x = self.fc(x.view(len(x), -1))
        x = self.softplus(x)
        return x

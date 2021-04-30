import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


# embedding layer
class Embedding(nn.Module):
    def __init__(self, input_dim, output_dim, item=False):
        super(Embedding, self).__init__()
        if item:
            self.embedding_table = nn.Embedding(input_dim, output_dim, padding_idx=0)
        else:
            self.embedding_table = nn.Embedding(input_dim, output_dim)

    def forward(self, x):
        return self.embedding_table(x)


class Intra_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(Intra_RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, lengths):
        x = self.dropout(x)
        gru_res, _ = self.gru(x, hidden)
        output = self.dropout(gru_res)
        output = self.linear(output)
        # the value in lengths is the index of the last element which is not padded 0
        idx = lengths.view(-1, 1, 1).expand(gru_res.size[0], 1, gru_res.size[2])
        # get rid of the padding zeros
        hidden_out = torch.gather(gru_res, 1, idx)
        hidden_out = hidden_out.squeeze().unsqueeze(0)
        return output, hidden_out


class Inter_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(Inter_RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x, hidden, idx):
        x = self.dropout(x)
        gru_res = self.gru(x, hidden)
        # Same as Intra_RNN
        hidden_idx = idx.view(-1, 1, 1).expand(gru_res.size(0), 1, gru_res.size(2))
        hidden_output = torch.gather(gru_res, 1, hidden_idx)
        hidden_output = hidden_output.squeeze().unsqueeze(0)
        hidden_output = self.dropout(hidden_output)
        return hidden_output

    def init_hidden(self, batch_size):
        return Variable(torch.zeros((1,batch_size,self.hidden_dim), dtype=torch.float))

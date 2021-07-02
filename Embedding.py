import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Net(nn.Module):
    def __init__(self, max_pj, embedding_dim, seq_len, n_actions, histo_len, rnn_hidden_dim, n_layers, rnn_dropout=0.0):
        super(Net, self).__init__()
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len  # state_1
        self.histo_len = histo_len # state_e
        self.n_actions = n_actions

        self.seq_embed = nn.Embedding(max_pj, embedding_dim, padding_idx=0)
        # TODO: self.histo_embed

        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=rnn_hidden_dim,
                           num_layers=n_layers, batch_first=True, dropout=rnn_dropout)
        self.h0 = torch.randn(n_layers, rnn_hidden_dim)
        self.c0 = torch.randn(n_layers, rnn_hidden_dim)

        self.pred_layer = nn.Linear(rnn_hidden_dim, n_actions, bias=True)

    def forward(self, state):
        seq = self.seq_embed(state[0: self.seq_len])
        batchsize = state.size()[0]
        h0 = torch.unsqueeze(self.h0, 1).repeat(1, batchsize, 1)
        c0 = torch.unsqueeze(self.c0, 1).repeat(1, batchsize, 1)

        # TODO: sort & pack & pad
        res, hn = self.rnn(seq, (h0, c0))   # res [bs, seq_len, hidden_dim]

        # TODO: histo normalization + concat ?

        actions_value = self.pred_layer(res)
        return actions_value

import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Net(nn.Module):
    def __init__(self, max_pj, embedding_dim, seq_len, max_actions, histo_len, rnn_hidden_dim, n_layers,
                 rnn_dropout=0.0):
        super(Net, self).__init__()
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len  # state_1
        self.histo_len = histo_len  # state_e
        self.max_actions = max_actions
        self.max_pros_len = 83
        self.cat_len = 10
        self.subcat_len = 35
        self.worker_len = 1820
        self.seq_layer_dim = 200
        self.task_layer_dim = 300
        self.cat_embedding_dim = 5
        self.sub_cat_embedding_dim = 10

        self.pro_embed = nn.Embedding(max_pj, embedding_dim, padding_idx=0)
        self.cat_embed = nn.Embedding(self.cat_len + 1, self.cat_embedding_dim, padding_idx=0)
        self.subcat_embed = nn.Embedding(self.subcat_len + 1, self.sub_cat_embedding_dim, padding_idx=0)
        self.worker_embed = nn.Embedding(self.worker_len+1, embedding_dim, padding_idx=0)

        # TODO: self.histo_embed

        self.rnn_pro = nn.LSTM(input_size=embedding_dim, hidden_size=rnn_hidden_dim,
                               num_layers=n_layers, batch_first=True, dropout=rnn_dropout)
        self.rnn_cat = nn.LSTM(input_size=self.cat_embedding_dim, hidden_size=rnn_hidden_dim,
                               num_layers=n_layers, batch_first=True, dropout=rnn_dropout)
        self.rnn_subcat = nn.LSTM(input_size=self.sub_cat_embedding_dim, hidden_size=rnn_hidden_dim,
                                  num_layers=n_layers, batch_first=True, dropout=rnn_dropout)

        self.h0 = torch.randn(n_layers, rnn_hidden_dim).cuda()
        self.c0 = torch.randn(n_layers, rnn_hidden_dim).cuda()
        self.h0_cat = torch.randn(n_layers, rnn_hidden_dim).cuda()
        self.c0_cat = torch.randn(n_layers, rnn_hidden_dim).cuda()
        self.h0_subcat = torch.randn(n_layers, rnn_hidden_dim).cuda()
        self.c0_subcat = torch.randn(n_layers, rnn_hidden_dim).cuda()
        # self.h0 = torch.randn(n_layers, rnn_hidden_dim)
        # self.c0 = torch.randn(n_layers, rnn_hidden_dim)
        # self.h0_cat = torch.randn(n_layers, rnn_hidden_dim)
        # self.c0_cat = torch.randn(n_layers, rnn_hidden_dim)
        # self.h0_subcat = torch.randn(n_layers, rnn_hidden_dim)
        # self.c0_subcat = torch.randn(n_layers, rnn_hidden_dim)
        self.layer1 = nn.Sequential(nn.Linear(rnn_hidden_dim, self.max_actions, bias=True), nn.ReLU(True))
        self.seq_layer = nn.Sequential(nn.Linear(rnn_hidden_dim * 3 + embedding_dim, self.seq_layer_dim, bias=True), nn.ReLU(True))
        self.task_layer = nn.Sequential(nn.Linear(
            (embedding_dim + self.cat_embedding_dim + self.sub_cat_embedding_dim) * (self.max_pros_len),
            self.task_layer_dim, bias=True), nn.ReLU(True))

        self.pred_layer = nn.Linear(self.seq_layer_dim + self.task_layer_dim, self.max_pros_len, bias=True)

    def forward(self, state):
        worker_embed = self.worker_embed(state[:, 0])
        current_index = self.seq_len + 1
        seq = self.pro_embed(state[:, 1: current_index])
        cate = self.cat_embed(state[:, current_index:current_index + self.seq_len])
        current_index += self.seq_len
        subcate = self.subcat_embed(state[:, current_index:current_index + self.seq_len])
        current_index += self.seq_len
        pro_state = state[:, current_index:-1:3]
        pro_embed = self.pro_embed(pro_state)
        cat_state = state[:, current_index + 1:-1:3]
        pro_cat_embed = self.cat_embed(cat_state)
        subcate_state = state[:, current_index+2:len(state[0]):3]
        pro_subcat_embed = self.subcat_embed(subcate_state)

        batchsize = state.size()[0]
        h0 = torch.unsqueeze(self.h0, 1).repeat(1, batchsize, 1)
        c0 = torch.unsqueeze(self.c0, 1).repeat(1, batchsize, 1)
        h0_cat = torch.unsqueeze(self.h0_cat, 1).repeat(1, batchsize, 1)
        c0_cat = torch.unsqueeze(self.c0_cat, 1).repeat(1, batchsize, 1)
        h0_subcat = torch.unsqueeze(self.h0_subcat, 1).repeat(1, batchsize, 1)
        c0_subcat = torch.unsqueeze(self.c0_subcat, 1).repeat(1, batchsize, 1)

        # TODO: sort & pack & pad
        res, (hn, cn) = self.rnn_pro(seq, (h0, c0))  # res [bs, seq_len, hidden_dim]
        res_cat, (hn_cat, cn_cat) = self.rnn_cat(cate, (h0_cat, c0_cat))  # res [bs, seq_len, hidden_dim]
        res_subcat, (hn_subcat, cn_subcat) = self.rnn_subcat(subcate, (h0_subcat, c0_subcat))  # res [bs, seq_len, hidden_dim]

        # TODO: histo normalization + concat ?
        # histo = state[:, self.seq_len:].float()
        # histo = torch.nn.functional.normalize(histo, dim=1)

        # todo: rnn 的res mean或者最后一个？ 目前取的是mean
        # res = torch.mean(res, dim=1)
        res = hn[-1, :, :]
        res_cat = hn_cat[-1, :, :]
        res_subcat = hn_subcat[-1, :, :]
        pros = torch.cat((pro_embed, pro_cat_embed, pro_subcat_embed), 2)
        pros = torch.reshape(pros, (batchsize,1, -1))
        pros = self.task_layer(pros)
        pros = torch.squeeze(pros, 1)
        history = torch.cat((worker_embed, res, res_cat, res_subcat), 1)
        history = self.seq_layer(history)
        res = torch.cat((history,pros),1)

        # res = torch.cat((res, histo), 1)
        actions_value = self.pred_layer(res)
        # actions_value = torch.nn.functional.softmax(actions_value)
        return actions_value

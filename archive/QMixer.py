import torch.nn.functional as F
import torch.nn as nn
import torch


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.fc_q = nn.Linear(64, 64)
        self.fc_k = nn.Linear(64, 64)
        self.fc_v = nn.Linear(64, 64)
        self.fc_o = nn.Linear(64, 64)

    def forward(self, x):
        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)

        q = q.view(q.size(0), -1, 8, 8)
        k = k.view(k.size(0), -1, 8, 8)
        v = v.view(v.size(0), -1, 8, 8)

        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)

        attention = torch.matmul(q, k.permute(0, 1, 3, 2))
        attention = F.softmax(attention, dim=-1)

        x = torch.matmul(attention, v)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), 64)

        x = self.fc_o(x)
        return x


class HypernetAtt(nn.Module):
    """
    mode='matrix' gets you a <n_agents x mixing_embed_dim> sized matrix
    mode='vector' gets you a <mixing_embed_dim> sized vector by averaging over agents
    mode='scalar' gets you a scalar by averaging over agents and embed dim
    ...per set of entities
    """

    def __init__(self, mode='matrix'):
        super(HypernetAtt, self).__init__()
        self.mha = MultiHeadAttention()
        self.mode = mode

    def forward(self, pair_enc):
        pair_enc_att = self.mha(pair_enc)

        if self.mode == 'vector':
            return pair_enc_att.mean(dim=0)
        elif self.mode == 'scalar':
            return pair_enc_att.mean(dim=(0, 1))

        return pair_enc_att


class QMixer(nn.Module):
    def __init__(self):
        super(QMixer, self).__init__()

        self.embed_dim = 64

        self.hyper_w_1 = HypernetAtt(mode='matrix')
        self.hyper_w_final = HypernetAtt(mode='vector')
        self.hyper_b_1 = HypernetAtt(mode='vector')
        self.V = HypernetAtt(mode='scalar')

    def forward(self, n_pair_enc, n_q_vals):
        # n_q_vals: (list) b x n_pairs
        # n_pair_enc: (list) b x n_pair x h*2

        Q_tot = []

        for pair_enc, q_vals in zip(n_pair_enc, n_q_vals):
            q_tot = self._forward(pair_enc, q_vals)
            Q_tot.append(q_tot)

        return Q_tot

    def _forward(self, pair_enc, q_vals):
        # first layer
        w1 = self.hyper_w_1(pair_enc)
        w1 = F.softmax(w1, dim=-1)
        b1 = self.hyper_b_1(pair_enc)

        q_vals = q_vals.view(1, 1, -1)
        hidden = F.elu(torch.bmm(q_vals, w1.unsqueeze(dim=0)) + b1)

        # final layer
        w_final = F.softmax(self.hyper_w_final(pair_enc), dim=-1)
        w_final = w_final.view(-1, self.embed_dim, 1)
        v = self.V(pair_enc).view(-1, 1, 1)

        q_tot = torch.bmm(hidden, w_final) + v

        return q_tot

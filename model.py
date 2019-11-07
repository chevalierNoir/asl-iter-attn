import torch
import alexnet
import math
import torch.nn as nn
import torch.nn.functional as functional


class AttnEncoder(nn.Module):
    def __init__(self, hidden_size, attn_size, output_size, n_layers, prior_gamma, cell="LSTM", pretrain=None):
        super(AttnEncoder, self).__init__()
        self.cell_type = cell
        self.conv = alexnet.alexnet(pretrain, output_size=1, p_2d=0.3)
        self.encoder_cell = AttnEncoderCell(hidden_size, attn_size, n_layers, prior_gamma, cell)
        self.lt = nn.Linear(hidden_size, output_size)
        self.classifier = nn.Softmax(dim=-1)

    def forward(self, Xs, h0, Ms=None):
        """
        Xs: (B, L, C, H, W), h0: (B, n_layer, n_hidden), Ms: (B, L, h, w)
        output: (B, L, V), (B, L, V), (B, F), (B, L, h, w)
        """
        if self.cell_type == 'LSTM':
            h0 = (h0[0].transpose(0, 1), h0[1].transpose(0, 1))
        else:
            h0 = h0.transpose(0, 1)
        xsz = list(Xs.size())
        Xs = Xs.view(*([-1] + xsz[2:]))
        Fs = self.conv(Xs)
        fsz = list(Fs.size())
        bsz, L_enc, hmap, wmap = xsz[0], xsz[1], fsz[2], fsz[3]
        Fs = Fs.transpose(1, 2).transpose(2, 3)
        Fs = Fs.view(*(xsz[:2]+[hmap*wmap]+[fsz[1]]))
        Fs = Fs.transpose(1, 0)
        if Ms is not None:
            Ms = Ms.view(*(xsz[:2] + [-1])).transpose(1, 0)
        ys, betas = [], []
        steps = Fs.size(0)
        h = h0
        for i in range(steps):
            Fi = Fs[i].transpose(0, 1).contiguous()
            if Ms is None:
                output, h, beta = self.encoder_cell(h, Fi)
            else:
                Mi = Ms[i].transpose(0, 1).contiguous()
                output, h, beta = self.encoder_cell(h, Fi, prior_map=Mi)
            ys.append(output)
            betas.append(beta)
        ys, betas = torch.stack(ys, 0), torch.stack(betas, 0)
        logits = self.lt(ys)
        probs = self.classifier(logits)
        betas = betas.transpose(0, 2).transpose(1, 2).contiguous()
        betas = betas.view(bsz*L_enc, -1).view(bsz*L_enc, hmap, wmap).view(bsz, L_enc, hmap, wmap)
        logits, probs = logits.transpose(0, 1), probs.transpose(0, 1)
        return logits, probs, h, betas



class AttnEncoderCell(nn.Module):
    def __init__(self, hidden_size, attn_size, n_layers, prior_gamma, cell="LSTM"):
        super(AttnEncoderCell, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.cell_type = cell
        if cell == "GRU":
            self.rnn_cell = nn.GRUCell(attn_size, hidden_size)
        elif cell == "LSTM":
            self.rnn_cell = nn.LSTMCell(attn_size, hidden_size)
        self.tanh = nn.Tanh()
        self.v = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.Wa = nn.Parameter(torch.zeros(attn_size, hidden_size))
        self.Wh = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.prior_gamma = prior_gamma
        # Initialization
        self.init_weight(self.v)

    def forward(self, hidden, attn, prior_map=None):
        # In: ([Layers, B, H], [Layers, B, H]), [N, B, A], [N, B, M], [N, B]
        # Out: [B, H], ([Layers, B, H], [Layers, B, H]), [N, B]
        prev_out = hidden[0][-1] if self.cell_type == "LSTM" else hidden[-1]
        N, B, A, H = attn.size()[0], attn.size()[1], attn.size()[2], self.hidden_size
        attn_weights = torch.matmul(attn.view(-1, A), self.Wa).view(N, B, H)+torch.matmul(prev_out, self.Wh)
        attn_weights = torch.matmul(self.tanh(attn_weights).view(-1, H), self.v).view(N, B)
        attn_weights = functional.softmax(attn_weights, dim=0)
        attn_weights = attn_weights*(prior_map.pow(self.prior_gamma)) if prior_map is not None else attn_weights
        s = (attn_weights.view(N, B, 1).repeat(1, 1, A) * attn).sum(dim=0)/attn_weights.sum(dim=0).view(B, 1).clamp(min=1.0e-5)  # [B, A]
        output = s
        hx, cx = [], []
        for i in range(self.n_layers):
            if self.cell_type == "GRU":
                h = self.rnn_cell(output, hidden[i])
                output = h
            else:
                h, c = self.rnn_cell(output, (hidden[0][i], hidden[1][i]))
                output = h
                cx.append(c)
            hx.append(h)
        hx = torch.stack(hx, 0)
        if self.cell_type == "GRU":
            return output, hx, attn_weights
        else:
            cx = torch.stack(cx, 0)
            return output, (hx, cx), attn_weights

    def init_weight(self, *args):
        for w in args:
            hin, hout = w.size()[0], w.size()[1]
            w.data.uniform_(-math.sqrt(6.0/(hin+hout)), math.sqrt(6.0/(hin+hout)))


def init_lstm_hidden(nlayer, batch_size, nhid, dtype=torch.float, device=torch.device('cuda')):
    return (torch.zeros((batch_size, nlayer, nhid), dtype=dtype, device=device),
            torch.zeros((batch_size, nlayer, nhid), dtype=dtype, device=device))

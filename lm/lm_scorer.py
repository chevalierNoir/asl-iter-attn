import sys
import torch
import torch.nn as nn
from collections import defaultdict
from rnn import RNN


class Scorer(object):
    def __init__(self, char_list, model_path, rnn_type, ninp, nhid, nlayers, device):
        char_list = list(char_list) + ['sil_start', 'sil_end']
        self.inv_vocab_map = dict([(i, c) for (i, c) in enumerate(char_list)])
        self.vocab_map = dict([(c, i) for (i, c) in enumerate(char_list)])
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.rnn = RNN(rnn_type, len(char_list), ninp, nhid, nlayers).to(self.device)
        self.rnn.load_state_dict(torch.load(model_path))
        self.rnn.eval()
        self.history = defaultdict(tuple)

    def get_score(self, string):
        if len(string) < 2:
            return 0, self.rnn.init_hidden(1)
        string_idx = map(lambda x: self.vocab_map[x], string)
        input = string_idx[:-1]
        grt = string_idx[1:]
        input, grt = torch.LongTensor(input).to(self.device), torch.LongTensor(grt).to(self.device)
        input = input.view(1, input.size()[0])
        init_hidden = self.rnn.init_hidden(1)
        pred, hidden = self.rnn(input, init_hidden)
        pred = pred.view(-1, pred.size(-1))
        loss = self.criterion(pred, grt)
        return -(len(string_idx)-1)*loss.item(), hidden

    def get_score_fast(self, strings):
        strings = [''.join(x) for x in strings]
        history_to_update = defaultdict(tuple)
        scores = []
        for string in strings:
            if len(string) <= 2:
                score, hidden_state = self.get_score(string)
                scores.append(score)
                history_to_update[string] = (score, hidden_state)
            elif string in self.history:
                history_to_update[string] = self.history[string]
                scores.append(self.history[string][0])
            elif string[:-1] in self.history:
                score, hidden = self.history[string[:-1]]
                input, grt = torch.LongTensor([self.vocab_map[string[-2]]]).view(1, 1).to(self.device), torch.LongTensor([self.vocab_map[string[-1]]]).to(self.device)
                pred, hidden = self.rnn(input, hidden)
                loss = self.criterion(pred.view(-1, pred.size(-1)), grt).item()
                history_to_update[string] = (score-loss, hidden)
                scores.append(score-loss)
            else:
                raise ValueError("%s not stored" % (string[:-1]))
        self.history = history_to_update
        return scores

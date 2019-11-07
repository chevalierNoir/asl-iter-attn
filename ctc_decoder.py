import math
import numpy as np


class Decoder(object):
    def __init__(self, labels, blank_index=0):
        self.labels = labels
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
        self.char_to_int = dict([(c, i) for (i, c) in enumerate(labels)])
        self.blank_index = blank_index
        space_index = len(labels)
        if ' ' in labels:
            space_index = labels.index(' ')
        self.space_index = space_index

    def greedy_decode(self, prob, digit=False):
        # prob: [seq_len, num_labels+1], numpy array
        indexes = np.argmax(prob, axis=1)
        string = []
        prev_index = -1
        for i in range(len(indexes)):
            if indexes[i] == self.blank_index:
                prev_index = -1
                continue
            elif indexes[i] == prev_index:
                continue
            else:
                if digit is False:
                    string.append(self.int_to_char[indexes[i]])
                else:
                    string.append(indexes[i])
                prev_index = indexes[i]
        return string

    def beam_decode(self, prob, beam_size, beta=0.0, gamma=0.0, scorer=None, digit=False):
        # prob: [seq_len, num_labels+1], numpy array
        # beta: lm coef, gamma: insertion coef
        seqlen = len(prob)
        beam_idx = np.argsort(prob[0, :])[-beam_size:].tolist()
        beam_prob = list(map(lambda x: math.log(prob[0, x]), beam_idx))
        beam_idx = list(map(lambda x: [x], beam_idx))
        for t in range(1, seqlen):
            topk_idx = np.argsort(prob[t, :])[-beam_size:].tolist()
            topk_prob = list(map(lambda x: prob[t, x], topk_idx))
            aug_beam_prob, aug_beam_idx = [], []
            for b in range(beam_size*beam_size):
                aug_beam_prob.append(beam_prob[b/beam_size])
                aug_beam_idx.append(list(beam_idx[b/beam_size]))
            # allocate
            for b in range(beam_size*beam_size):
                i, j = b/beam_size, b % beam_size
                aug_beam_idx[b].append(topk_idx[j])
                aug_beam_prob[b] = aug_beam_prob[b]+math.log(topk_prob[j])
            # merge
            merge_beam_idx, merge_beam_prob = [], []
            for b in range(beam_size*beam_size):
                if aug_beam_idx[b][-1] == aug_beam_idx[b][-2]:
                    beam, beam_prob = aug_beam_idx[b][:-1], aug_beam_prob[b]
                elif aug_beam_idx[b][-2] == self.blank_index:
                    beam, beam_prob = aug_beam_idx[b][:-2]+[aug_beam_idx[b][-1]], aug_beam_prob[b]
                else:
                    beam, beam_prob = aug_beam_idx[b], aug_beam_prob[b]
                beam_str = list(map(lambda x: self.int_to_char[x], beam))
                if beam_str not in merge_beam_idx:
                    merge_beam_idx.append(beam_str)
                    merge_beam_prob.append(beam_prob)
                else:
                    idx = merge_beam_idx.index(beam_str)
                    merge_beam_prob[idx] = np.logaddexp(merge_beam_prob[idx], beam_prob)

            if scorer is not None:
                merge_beam_prob_lm, ins_bonus, strings = [], [], []
                for b in range(len(merge_beam_prob)):
                    if merge_beam_idx[b][-1] == self.int_to_char[self.blank_index]:
                        strings.append(merge_beam_idx[b][:-1])
                        ins_bonus.append(len(merge_beam_idx[b][:-1]))
                    else:
                        strings.append(merge_beam_idx[b])
                        ins_bonus.append(len(merge_beam_idx[b]))
                lm_scores = scorer.get_score_fast(strings)
                for b in range(len(merge_beam_prob)):
                    total_score = merge_beam_prob[b]+beta*lm_scores[b]+gamma*ins_bonus[b]
                    merge_beam_prob_lm.append(total_score)

            if scorer is None:
                ntopk_idx = np.argsort(np.array(merge_beam_prob))[-beam_size:].tolist()
            else:
                ntopk_idx = np.argsort(np.array(merge_beam_prob_lm))[-beam_size:].tolist()
            beam_idx = list(map(lambda x: merge_beam_idx[x], ntopk_idx))
            for b in range(len(beam_idx)):
                beam_idx[b] = list(map(lambda x: self.char_to_int[x], beam_idx[b]))
            beam_prob = list(map(lambda x: merge_beam_prob[x], ntopk_idx))
        if self.blank_index in beam_idx[-1]:
            pred = beam_idx[-1][:-1]
        else:
            pred = beam_idx[-1]
        if digit is False:
            pred = list(map(lambda x: self.int_to_char[x], pred))
        return pred

import torch
import numpy as np

def get_ctc_vocab(char_list):
    # blank
    ctc_char_list = "_" + char_list
    ctc_map, inv_ctc_map = {}, {}
    for i, char in enumerate(ctc_char_list):
        ctc_map[char] = i
        inv_ctc_map[i] = char
    return ctc_map, inv_ctc_map, ctc_char_list


def get_corpus(corpus_file):
    with open(corpus_file, "r") as fo:
        lns = list(map(lambda x: x.strip().split(',')[1], fo.readlines()))
        corpus = lns
    return corpus


def numerize(sents, vocab_map):
    outs = []
    for sent in sents:
        outs.append(list(map(lambda x: vocab_map[x], sent)))
    return outs


def get_onehot(idx, num_vocab):
    onehot = np.zeros(len(idx), num_vocab)
    onehot[range(len(idx)), idx] = 1.0
    return onehot


def make_batch(sents, pad_id):
    batch_size = len(sents)
    sents_len = list(map(len, sents))
    mask, sents_pad = np.zeros((len(sents), max(sents_len)), dtype=np.float32), np.zeros((len(sents), max(sents_len)), dtype=np.int32)
    for i, sent_len in enumerate(sents_len):
        mask[i, :sent_len-1] = 1
        sents_pad[i, :sent_len] = sents[i]
    return sents_pad, mask


def get_loss(pred, grt, mask):
    # pred: [B, L, V], grt: [B, L], mask: [B, L]
    pred = pred.clamp(min=1.0e-5, max=1-1.0e-5)
    pred = -torch.log(pred)
    bsz, L, V = pred.size(0), pred.size(1), pred.size(2)
    pred, grt, mask = pred.view(bsz*L, V), grt.view(-1), mask.view(-1)
    grt_onehot = pred.new_zeros(bsz*L, V)
    grt_onehot[torch.arange(bsz*L, dtype=torch.long), grt] = 1
    l = ((grt_onehot * pred).sum(dim=-1) * mask).sum() / mask.sum()
    return l

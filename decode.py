import os
import lev
import argparse
import torch
import configparser
import numpy as np
import scipy.io as sio
from ctc_decoder import Decoder
from lm import utils
from lm.lm_scorer import Scorer as Scorer
from collections import OrderedDict


def parse_csv(csv_file):
    with open(csv_file, "r") as fo:
        lns = map(lambda x: x.strip(), fo.readlines())
    labels = OrderedDict()
    for ln in lns:
        subdir, label, _ = ln.split(",")
        labels[subdir] = label
    return labels


def main():
    parser = argparse.ArgumentParser(description="decode")
    parser.add_argument("--csv", type=str, help="csv file")
    parser.add_argument("--pred", type=str, help="predict dir")
    parser.add_argument("--conf", type=str, help="config file")
    parser.add_argument("--decode_type", type=str, help="option: greedy|beam")
    parser.add_argument("--beam_size", type=int, help="beam size")
    parser.add_argument("--lm_beta", type=float, default=0.1, help="lm beta")
    parser.add_argument("--lm_pth", type=str, help="lm model path")
    parser.add_argument("--ins_gamma", type=float, default=0.0, help="insertion gamma")
    args = parser.parse_args()
    labels = parse_csv(args.csv)
    config = configparser.ConfigParser()
    config.read(args.conf)
    char_list = '_' + config['LANG']['chars']
    decoder = Decoder(char_list, blank_index=0)
    vocab_map, inv_vocab_map = decoder.char_to_int, decoder.int_to_char
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if args.decode_type == "beam":
        print("beam size: %d, lm beta: %.3f, ins gamma: %.3f" % (args.beam_size, args.lm_beta, args.ins_gamma))
        scorer = Scorer(config['LANG']['chars'], args.lm_pth, rnn_type=config['LM']['rnn_type'], ninp=config['LM'].getint('ninp'), nhid=config['LM'].getint('nhid'), nlayers=config['LM'].getint('nlayers'), device=device)
    else:
        print("greedy")
    pred_arr, lb_arr = [], []
    for subdir, label in labels.items():
        lb_arr.append(np.array(map(lambda x: vocab_map[x], label)))
        prob = sio.loadmat(os.path.join(args.pred, subdir, "prob.mat"))["prob"]
        if args.decode_type == "greedy":
            pred = decoder.greedy_decode(prob, digit=True)
        elif args.decode_type == "beam":
            pred = decoder.beam_decode(prob, beam_size=args.beam_size, beta=args.lm_beta, gamma=args.ins_gamma, scorer=scorer, digit=True)
        else:
            raise ValueError("Option for decode_type: greedy|beam")
        pred_arr.append(np.array(pred))
    lev_acc = lev.compute_acc(pred_arr, lb_arr)
    print("Accuracy: %.3f" % (lev_acc))
    return


if __name__ == "__main__":
    main()

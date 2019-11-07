import os
import torch
import utils
import argparse
import configparser
import numpy as np
from torch import optim
from torch import nn
from torch.autograd import Variable
from rnn import RNN

np.random.seed(222)
torch.manual_seed(222)
torch.cuda.manual_seed(222)

def train(rnn, optimizer, train_data, batch_size, vocab_map, device):
    rnn.train()
    perm = np.random.permutation(range(len(train_data)))
    l = []
    for i in range(0, len(perm), batch_size):
        optimizer.zero_grad()
        input, grt = [], []
        for j in perm[i: i+batch_size]:
            input.append([vocab_map["sil_start"]]+train_data[perm[j]]+[vocab_map["sil_end"]])
            grt.append(input[-1][1:]+[vocab_map["sil_end"]])
        input, mask = utils.make_batch(input, vocab_map['sil_end'])
        grt, _ = utils.make_batch(grt, vocab_map['sil_end'])
        input, grt, mask = torch.LongTensor(input).to(device), torch.LongTensor(grt).to(device), torch.FloatTensor(mask).to(device)
        init_hidden = rnn.init_hidden(len(input))
        init_hidden = (init_hidden[0].to(device), init_hidden[1].to(device))
        pred, _ = rnn(input, init_hidden)
        loss = utils.get_loss(pred, grt, mask)
        loss.backward()
        optimizer.step()
        l.append(loss.item())
    l = sum(l) / len(l)
    return l


def test(rnn, test_data, vocab_map, device):
    rnn.eval()
    l = []
    test_batch_size = 1
    perm = range(len(test_data))
    for i in range(0, len(perm), test_batch_size):
        input, grt = [], []
        for j in perm[i: i+test_batch_size]:
            input.append([vocab_map["sil_start"]]+test_data[perm[j]]+[vocab_map["sil_end"]])
            grt.append(input[-1][1:]+[vocab_map["sil_end"]])
        input, mask = utils.make_batch(input, vocab_map['sil_end'])
        grt, _ = utils.make_batch(grt, vocab_map['sil_end'])
        with torch.no_grad():
            input, grt, mask = torch.LongTensor(input).to(device), torch.LongTensor(grt).to(device), torch.FloatTensor(mask).to(device)
            init_hidden = rnn.init_hidden(test_batch_size)
            init_hidden = (init_hidden[0].to(device), init_hidden[1].to(device))
            pred, _ = rnn(input, init_hidden)
            loss = utils.get_loss(pred, grt, mask)
            l.append(loss.item())
    l = sum(l) / len(l)
    return l

def main():
    parser = argparse.ArgumentParser(description="Language Model")
    parser.add_argument("--output", type=str, help="output dir")
    parser.add_argument("--train", type=str, help="train csv file")
    parser.add_argument("--dev", type=str, help="dev csv file")
    parser.add_argument("--conf", type=str, help="config file")
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    log_file, best_model, last_model = os.path.join(args.output, 'log'), os.path.join(args.output, 'best.pth'), os.path.join(args.output, 'latest.pth')
    config = configparser.ConfigParser()
    config.read(args.conf)
    char_list = list(config['LANG']['chars']) + ['sil_start', 'sil_end']
    vocab_map = dict([(c, i) for (i, c) in enumerate(char_list)])
    train_data, dev_data = utils.get_corpus(args.train), utils.get_corpus(args.dev)
    train_data, dev_data = utils.numerize(train_data, vocab_map), utils.numerize(dev_data, vocab_map)
    ntoken = len(vocab_map)
    ninp, nhid, nlayers = config['LM'].getint('ninp'), config['LM'].getint('nhid'), config['LM'].getint('nlayers')
    rnn_type = config['LM']['rnn_type']
    learning_rate = config['LM'].getfloat('learning_rate')
    epochs = config['LM'].getint('epochs')
    batch_size = config['LM'].getint('batch_size')
    device = torch.device('cuda')
    rnn = RNN(rnn_type, ntoken, ninp, nhid, nlayers)
    optimizer = optim.SGD(rnn.parameters(), learning_rate)
    rnn.to(device)

    min_dev_loss = float('inf')
    for ep in range(epochs):
        train_loss = train(rnn, optimizer, train_data, batch_size, vocab_map, device)
        dev_loss = test(rnn, dev_data, vocab_map, device)
        pcont = 'Epoch %d, train loss: %.3f, dev loss: %.3f' % (ep, train_loss, dev_loss)
        print(pcont)
        with open(log_file, 'a+') as fo:
            fo.write(pcont+'\n')
        torch.save(rnn.state_dict(), last_model)
        if dev_loss < min_dev_loss:
            min_dev_loss = dev_loss
            torch.save(rnn.state_dict(), best_model)

if __name__ == '__main__':
    main()

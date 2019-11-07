from __future__ import print_function
from __future__ import division

import os
import sys
import argparse
import configparser
import torch
import pickle
import lev
import random
import string
import numpy as np
import torch.utils.data as tud
import torch.optim as optim
from ctc_decoder import Decoder
from lm import utils
from torch import nn
from warpctc_pytorch import CTCLoss
from model import AttnEncoder, init_lstm_hidden
import data as dataset
from torchvision import transforms

np.random.seed(222)
torch.manual_seed(222)
torch.cuda.manual_seed(222)


def train(encoder, train_loader, clip, hypers, cnn_optimizer, lstm_optimizer, ctc_loss, decoder, log_path, model_path, hyper_path, device, interval):
    encoder.train()
    hidden_size, n_layers = encoder.encoder_cell.hidden_size, encoder.encoder_cell.n_layers
    larr, pred_arr, label_arr = [], [], []
    for i_batch, sample in enumerate(train_loader):
        cnn_optimizer.zero_grad()
        lstm_optimizer.zero_grad()
        imgs, priors, labels, prob_sizes, label_sizes = sample['image'], sample['prior'], sample['label'], sample['prob_size'], sample['label_size']
        h0 = init_lstm_hidden(n_layers, len(imgs), hidden_size, device=device)
        logits, probs, _, _ = encoder(imgs, h0, priors)
        logits, probs = logits.transpose(0, 1), probs.transpose(0, 1)

        bsz = len(imgs)
        l = ctc_loss(logits, labels, prob_sizes, label_sizes) / bsz
        l.backward()
        torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
        cnn_optimizer.step()
        lstm_optimizer.step()
        larr.append(l.item())
        # Decode
        probs = probs.transpose(0, 1).cpu().data.numpy()
        for j in range(len(probs)):
            pred = decoder.greedy_decode(probs[j], digit=True)
            pred_arr.append(pred)
            start, end = sum(label_sizes[:j]), sum(label_sizes[:j+1])
            label_arr.append(labels[start: end].tolist())
        hypers['step'] += 1
        if hypers['step'] % interval == 0:
            acc = lev.compute_acc(pred_arr, label_arr)
            l = sum(larr)/len(larr)
            pcont = "Step %d, train loss: %.3f, acc (LEV): %.3f" % (hypers['step'], l, acc)
            print(pcont)
            with open(log_path, 'a+') as fo:
                fo.write(pcont+"\n")
            with open(model_path, 'wb') as fo:
                torch.save(encoder.state_dict(), fo)
            with open(hyper_path, 'wb') as fo:
                pickle.dump(hypers, fo)
            larr, pred_arr, label_arr = [], [], []
    return

def evaluate(encoder, loader, ctc_loss, decoder, device):
    encoder.eval()
    hidden_size, n_layers = encoder.encoder_cell.hidden_size, encoder.encoder_cell.n_layers
    larr, pred_arr, label_arr = [], [], []
    for i_batch, sample in enumerate(loader):
        imgs, priors, labels, prob_sizes, label_sizes = sample['image'], sample['prior'], sample['label'], sample['prob_size'], sample['label_size']
        with torch.no_grad():
            h0 = init_lstm_hidden(n_layers, len(imgs), hidden_size, device=device)
            logits, probs, _, _ = encoder(imgs, h0, priors)
            logits, probs = logits.transpose(0, 1), probs.transpose(0, 1)

        bsz = len(imgs)
        l = ctc_loss(logits, labels, prob_sizes, label_sizes)/bsz
        larr.append(l.item())

        # Decode
        probs = probs.transpose(0, 1).cpu().data.numpy()
        for j in range(len(probs)):
            pred = decoder.greedy_decode(probs[j], digit=True)
            pred_arr.append(pred)
            start, end = sum(label_sizes[:j]), sum(label_sizes[:j+1])
            label_arr.append(labels[start: end].tolist())
    acc = lev.compute_acc(pred_arr, label_arr)
    l = sum(larr)/len(larr)
    return l, acc

def main():
    parser = argparse.ArgumentParser(description="Attn Encoder")
    parser.add_argument("--img", type=str, help="image dir")
    parser.add_argument("--prior", type=str, help="prior dir")
    parser.add_argument("--csv", type=str, help="csv dir")
    parser.add_argument("--conf", type=str, help="config file")
    parser.add_argument("--output", type=str, help="output dir")
    parser.add_argument("--pretrain", type=str, default=None, help="pretrain path")
    parser.add_argument("--cont", action="store_true", help="continue training")
    parser.add_argument("--epoch", type=int, default=1, help="epoch")
    parser.add_argument("--optim_step_size", type=int, default=30, help="lr decay step size")
    parser.add_argument("--optim_gamma", type=float, default=0.1, help="lr decay rate")
    parser.add_argument("--scaling", action="store_true", help="data augmentation (scaling)")
    parser.add_argument("--img_scale", type=float, default=1., nargs="+", help="image scales")
    parser.add_argument("--map_scale", type=int, default=13, nargs="+", help="map scales")
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    best_path = os.path.join(args.output, "best.pth")
    latest_path = os.path.join(args.output, "latest.pth")
    log = os.path.join(args.output, "log")
    hyper_path = os.path.join(args.output, "hyper.pth")

    config = configparser.ConfigParser()
    config.read(args.conf)
    model_cfg, lang_cfg, img_cfg = config['MODEL'], config['LANG'], config['IMAGE']
    hidden_size, attn_size, n_layers = model_cfg.getint('hidden_size'), model_cfg.getint('attn_size'), model_cfg.getint('n_layers')
    prior_gamma = model_cfg.getfloat('prior_gamma')
    learning_rate = model_cfg.getfloat('learning_rate')
    batch_size = model_cfg.getint('batch_size')
    char_list = lang_cfg['chars'] # " '&.@acbedgfihkjmlonqpsrutwvyxz"
    immean, imstd = [float(x) for x in config['IMAGE']['immean'].split(',')], [float(x) for x in config['IMAGE']['imstd'].split(',')] # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    upper_len = model_cfg.getint('upper_length')
    clip = model_cfg.getfloat('clip')
    save_interval = model_cfg.getint('interval')
    epochs = args.epoch
    optim_step_size, optim_gamma = args.optim_step_size, args.optim_gamma

    train_csv, dev_csv = os.path.join(args.csv, 'train.csv'), os.path.join(args.csv, 'dev.csv')

    device, cpu = torch.device('cuda'), torch.device('cpu')

    vocab_map, inv_vocab_map, char_list = utils.get_ctc_vocab(char_list)

    if type(args.img_scale) == list and type(args.map_scale) == list:
        scale_range, hw_range = args.img_scale, [(x, x) for x in args.map_scale]
    elif type(args.img_scale) == float and type(args.map_scale) == int:
        scale_range, hw_range = [args.img_scale], [(args.map_scale, args.map_scale)]
    else:
        raise AttributeError('scale: list or float/int')

    if not args.scaling:
        tsfm_train = transforms.Compose([dataset.ToTensor(device), dataset.Rescale(scale_range, hw_range, origin_scale=True), dataset.Normalize(immean, imstd, device)])
        tsfm_test = transforms.Compose([dataset.ToTensor(device), dataset.Rescale(scale_range, hw_range, origin_scale=True), dataset.Normalize(immean, imstd, device)])
    else:
        # scale_range = [1] # [1, 0.8, 1.2] # [1, 0.8]
        # hw_range = [(13, 13)]  # [(13, 13), (10, 10), (15, 15)] # [(13, 13), (10, 10)]
        tsfm_train = transforms.Compose([dataset.ToTensor(device), dataset.Rescale(scale_range, hw_range), dataset.Normalize(immean, imstd, device)])
        tsfm_test = transforms.Compose([dataset.ToTensor(device), dataset.Rescale(scale_range, hw_range, origin_scale=True), dataset.Normalize(immean, imstd, device)])

    sld_train_data = dataset.SLData(args.img, args.prior, train_csv, vocab_map, transform=tsfm_train, upper_len=upper_len)
    sld_dev_data = dataset.SLData(args.img, args.prior, dev_csv, vocab_map, transform=tsfm_test, upper_len=float('inf')) # dataset.Rescale([1], [(13, 13)])

    encoder = AttnEncoder(hidden_size=hidden_size, attn_size=attn_size,
                          output_size=len(char_list), n_layers=n_layers,
                          prior_gamma=prior_gamma, pretrain=args.pretrain)
    encoder.to(device)
    if torch.cuda.device_count() > 1:
        print('Using %d GPUs' % (torch.cuda.device_count()))
        encoder = nn.DataParallel(encoder)
    hypers = {'step': 0, 'epoch': 0, 'best_dev_acc': -1, 'perm': np.random.permutation(len(sld_train_data)).tolist()}

    if args.cont:
        print("Load %s, %s" % (latest_path, hyper_path))
        encoder.load_state_dict(torch.load(latest_path))
        try:
            with open(hyper_path, 'rb') as fo:
                hypers = pickle.load(fo)
        except Exception as err:
            print("Error loading %s: %s" % (hyper_path, err))
            hypers = {'step': 0, 'epoch': 0, 'best_dev_acc': -1, 'perm': np.random.permutation(len(sld_train_data)).tolist()}

    train_loader = tud.DataLoader(sld_train_data, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn_ctc)
    dev_loader = tud.DataLoader(sld_dev_data, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn_ctc)

    print('Optimizer, decay %.5f after %d epochs' % (optim_gamma, optim_step_size))
    cnn_optimizer = optim.SGD(encoder.conv.parameters(), lr=learning_rate)
    lstm_optimizer = optim.SGD(list(encoder.encoder_cell.parameters())+list(encoder.lt.parameters()), lr=learning_rate)
    cnn_scheduler = optim.lr_scheduler.StepLR(cnn_optimizer, step_size=optim_step_size, gamma=optim_gamma)
    lstm_scheduler = optim.lr_scheduler.StepLR(lstm_optimizer, step_size=optim_step_size, gamma=optim_gamma)

    decoder = Decoder(char_list)
    ctc_loss = CTCLoss() # normalize over batch

    print('%d training epochs' % (epochs))
    for ep in range(epochs):
        cnn_scheduler.step()
        lstm_scheduler.step()
        if ep < hypers['epoch']:
            continue
        for p in cnn_optimizer.param_groups:
            print('CNN', p['lr'])
        for p in lstm_optimizer.param_groups:
            print('LSTM', p['lr'])
        train(encoder, train_loader, clip, hypers, cnn_optimizer, lstm_optimizer, ctc_loss, decoder, log, latest_path, hyper_path, device, save_interval)

        dl, dacc = evaluate(encoder, dev_loader, ctc_loss, decoder, device)
        pcont = 'Epoch %d, dev loss: %.3f, dev acc (LEV): %.3f' % (ep, dl, dacc)
        print(pcont)
        with open(log, 'a+') as fo:
            fo.write(pcont+"\n")
        # save model and hyperparameter setting
        hypers['epoch'] = ep
        if hypers['best_dev_acc'] < dacc:
            hypers['best_dev_acc'] = dacc
            with open(best_path, 'wb') as fo:
                torch.save(encoder.state_dict(), fo)
        with open(hyper_path, 'wb') as fo:
            pickle.dump(hypers, fo)
    return

if __name__ == '__main__':
    main()

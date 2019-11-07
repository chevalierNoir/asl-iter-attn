from __future__ import print_function
from __future__ import division

import os
import sys
import argparse
import torch
import configparser
import scipy.io as sio
import torch.utils.data as tud
import data as dataset
from lm import utils
from torch import nn
from model import AttnEncoder, init_lstm_hidden
from torchvision import transforms


def get_beta(encoder, loaders, output_dir, device):
    encoder.eval()
    hidden_size, n_layers = encoder.encoder_cell.hidden_size, encoder.encoder_cell.n_layers
    for loader in loaders:
        for i_batch, sample in enumerate(loader):
            imgs, priors, labels, prob_sizes, label_sizes = sample['image'], sample['prior'], sample['label'], sample['prob_size'], sample['label_size']
            with torch.no_grad():
                h0 = init_lstm_hidden(n_layers, len(imgs), hidden_size, device=device)
                _, _, _, betas = encoder(imgs, h0, priors)
            betas = betas.cpu().numpy()
            for j in range(len(betas)):
                beta_full = os.path.join(output_dir, sample['imdir'][j])
                if not os.path.isdir(beta_full):
                    os.makedirs(beta_full)
                beta_full = os.path.join(beta_full, 'beta.mat')
                sio.savemat(beta_full, {'beta': betas[j], 'hw': (imgs.size(-2), imgs.size(-1))})
    return

def get_prob(encoder, loader, output_dir, device):
    encoder.eval()
    hidden_size, n_layers = encoder.encoder_cell.hidden_size, encoder.encoder_cell.n_layers
    for i_batch, sample in enumerate(loader):
        imgs, priors, labels, prob_sizes, label_sizes = sample['image'], sample['prior'], sample['label'], sample['prob_size'], sample['label_size']
        with torch.no_grad():
            h0 = init_lstm_hidden(n_layers, len(imgs), hidden_size, device=device)
            _, probs, _, _ = encoder(imgs, h0, priors)
        probs = probs.cpu().numpy()
        for j in range(len(probs)):
            prob_full = os.path.join(output_dir, sample['imdir'][j])
            if not os.path.isdir(prob_full):
                os.makedirs(prob_full)
            prob_full = os.path.join(prob_full, 'prob.mat')
            sio.savemat(prob_full, {'prob': probs[j]})
    return

def main():
    parser = argparse.ArgumentParser(description="Attn Encoder")
    parser.add_argument("--img", type=str, help="image dir")
    parser.add_argument("--prior", type=str, help="prior dir")
    parser.add_argument("--csv", type=str, help="csv dir")
    parser.add_argument("--conf", type=str, help="config file")
    parser.add_argument("--output", type=str, help="output dir")
    parser.add_argument("--model", type=str, help="model path")
    parser.add_argument("--partition", type=str, help="train|dev|test")
    parser.add_argument("--task", type=str, help="beta|prob")
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    config = configparser.ConfigParser()
    config.read(args.conf)
    model_cfg, lang_cfg, img_cfg = config['MODEL'], config['LANG'], config['IMAGE']
    hidden_size, attn_size, n_layers = model_cfg.getint('hidden_size'), model_cfg.getint('attn_size'), model_cfg.getint('n_layers')
    prior_gamma = model_cfg.getfloat('prior_gamma')
    batch_size = 1
    char_list = lang_cfg['chars']
    immean, imstd = [float(x) for x in config['IMAGE']['immean'].split(',')], [float(x) for x in config['IMAGE']['imstd'].split(',')]
    train_csv, dev_csv, test_csv = os.path.join(args.csv, 'train.csv'), os.path.join(args.csv, 'dev.csv'), os.path.join(args.csv, 'test.csv')

    device, cpu = torch.device('cuda'), torch.device('cpu')

    vocab_map, inv_vocab_map, char_list = utils.get_ctc_vocab(char_list)

    encoder = AttnEncoder(hidden_size=hidden_size, attn_size=attn_size,
                          output_size=len(char_list), n_layers=n_layers,
                          prior_gamma=prior_gamma, pretrain=None)
    encoder.to(device)
    if torch.cuda.device_count() > 1:
        print('Using %d GPUs' % (torch.cuda.device_count()))
        encoder = nn.DataParallel(encoder)

    print('Load model: %s' % (args.model))
    encoder.load_state_dict(torch.load(args.model))

    scale_range = [0]
    hw_range = [(0, 0)] 
    tsfm = transforms.Compose([dataset.ToTensor(device), dataset.Rescale(scale_range, hw_range, origin_scale=True), dataset.Normalize(immean, imstd, device)])

    train_data = dataset.SLData(args.img, args.prior, train_csv, vocab_map, transform=tsfm, upper_len=float('inf'))
    dev_data = dataset.SLData(args.img, args.prior, dev_csv, vocab_map, transform=tsfm, upper_len=float('inf'))
    test_data = dataset.SLData(args.img, args.prior, test_csv, vocab_map, transform=tsfm, upper_len=float('inf'))

    train_loader = tud.DataLoader(train_data, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn_ctc)
    dev_loader = tud.DataLoader(dev_data, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn_ctc)
    test_loader = tud.DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn_ctc)

    if args.task == 'beta':
        get_beta(encoder, [train_loader, dev_loader, test_loader], args.output, device)
    elif args.task == 'prob':
        if args.partition == 'train':
            loader = train_loader
        elif args.partition == 'dev':
            loader = dev_loader
        elif args.partition == 'test':
            loader = test_loader
        else:
            raise ValueError('partition: train|dev|test')
        get_prob(encoder, loader, args.output, device)
    return


if __name__ == "__main__":
    main()

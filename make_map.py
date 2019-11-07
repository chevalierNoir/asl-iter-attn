import os
import cv2
import torch
import alexnet
import numpy as np
import scipy.io as sio
import argparse
from alexnet import alexnet

parser = argparse.ArgumentParser(description="image to map")
parser.add_argument("--img", type=str, help="input prior image dir")
parser.add_argument("--mat", type=str, help="output mat dir")
parser.add_argument("--csv", type=str, help="csv file")
parser.add_argument("--device", type=str, default='cuda', help="cpu/cuda")
parser.add_argument("--start", type=int, help="start id")
parser.add_argument("--end", type=int, help="end id")
parser.add_argument("--size", type=int, help="image w/h")
args = parser.parse_args()

def idx2name(idx):
    return "0"*(4-len(str(idx)))+str(idx)

def img2mat(in_dir, out_dir, csv, start, end, device, size=None):
    with open(csv, "r") as fo:
        lns = fo.readlines()
        subdirs, nums_frames = [], []
        for ln in lns:
            subdir, _, num_frame = ln.split(",")
            subdirs.append(subdir)
            nums_frames.append(int(num_frame))
    end = len(subdirs) if end == -1 else end
    cnn = alexnet(output_size=1).to(torch.device(device))
    for i, subdir in enumerate(subdirs):
        if not (i >= start and i < end):
            continue
        num_frame = nums_frames[i]
        imnames = [idx2name(i)+".jpg" for i in range(1, num_frame+1)]
        priors = []
        if size is None:
            img = cv2.imread(os.path.join(in_dir, subdir, imnames[0])).astype(np.float32).transpose((2, 0, 1))
            img = torch.from_numpy(img).unsqueeze(dim=0)
            with torch.no_grad():
                fmap = cnn(img.to(device))
            fh, fw = fmap.size(2), fmap.size(3)
        else:
            fh, fw = size, size
        for imname in imnames:
            img = cv2.imread(os.path.join(in_dir, subdir, imname), 0)
            prior = cv2.resize(img, (fw, fh)).astype(np.float32)
            prior = prior/max(1, prior.max())
            priors.append(prior)
        priors = np.stack(priors)
        prior_dir = os.path.join(out_dir, subdir)
        if not os.path.isdir(prior_dir):
            os.makedirs(prior_dir)
        np.save(os.path.join(prior_dir, "prior.npy"), priors)
    return


if __name__ == "__main__":
    img2mat(args.img, args.mat, args.csv, args.start, args.end, args.device, args.size)

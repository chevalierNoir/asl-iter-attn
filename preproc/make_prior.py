# Make prior map
from __future__ import print_function
from __future__ import division

import os
import argparse
import cv2
import numpy as np
import scipy.io as sio


def main():
    parser = argparse.ArgumentParser(description="prior map")
    parser.add_argument("--OPT", type=str, help="opt dir")
    parser.add_argument("--PRIOR", type=str, help="out dir")
    parser.add_argument("--csv", type=str, help="csv file")
    parser.add_argument("--start", type=int, help="start id")
    parser.add_argument("--end", type=int, help="end id")
    args = parser.parse_args()
    # make prior in image form
    window_size = 3
    img_suffix = '.jpg'
    subdirs, num_frames = [], []
    lns = open(args.csv, 'r').readlines()
    for ln in lns:
        subdir, _, num_frame = ln.strip().split(',')
        subdirs.append(subdir)
        num_frames.append(int(num_frame))
    start, end = args.start, len(subdirs) if args.end == -1 else args.end
    for sid, subdir in enumerate(subdirs):
        if not (sid >= start and sid < end):
            continue
        opt_dir, out_dir = os.path.join(args.OPT, subdir), os.path.join(args.PRIOR, subdir)
        imnames = ["0"*(4-len(str(i))) + str(i) + img_suffix for i in range(1, num_frames[sid]+1)]
        opt_imgs = list(map(lambda x: cv2.imread(os.path.join(opt_dir, x), 0), imnames))
        for i, opt_img in enumerate(opt_imgs):
            opt_img = (opt_img/float(max(opt_img.max(), 1))).astype(np.float32)
            if opt_img.sum() == 0:
                opt_img = np.zeros_like(opt_img)
            opt_imgs[i] = opt_img
        opt_imgs = [np.zeros_like(opt_imgs[0]) for _ in range(window_size//2)] + opt_imgs + \
                   [np.zeros_like(opt_imgs[0]) for _ in range(window_size//2)]
        opt_imgs = np.stack(opt_imgs, axis=0)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        for i, imname in enumerate(imnames):
            opt_img = np.mean(opt_imgs[i: i+window_size], axis=0)
            score_img = opt_img
            score_img = (255*score_img).astype(np.int32)
            score_imname = os.path.join(out_dir, imname)
            cv2.imwrite(score_imname, score_img)
        print(subdir)
    return 0


if __name__ == "__main__":
    main()

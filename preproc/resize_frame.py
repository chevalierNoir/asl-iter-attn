from __future__ import division
from __future__ import print_function

import os
import cv2
import json
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='resize frame')
    parser.add_argument('--input_dir', type=str, help='dir of input image')
    parser.add_argument('--output_dir', type=str, help='dir of output image')
    parser.add_argument('--lambda_dir', type=str, help='lambda dir')
    parser.add_argument('--csv', type=str, help='csv file')
    parser.add_argument('--df', type=str, help='csv file')
    parser.add_argument('--start', type=int, help='start id')
    parser.add_argument('--end', type=int, help='end id')
    parser.add_argument('--rtype', type=str, default='aspect', help='resize type (naive|aspect)')
    args = parser.parse_args()

    ## resize image
    wt, ht = 224, 224
    thr = 20 # diff between target area and current area
    round_size = 32
    csv_file = args.csv
    in_dir = args.input_dir
    out_dir = args.output_dir
    lambda_dir = args.lambda_dir
    targets = list(map(lambda x: x.strip().split(',')[0], open(args.csv, 'r').readlines()))
    start = args.start
    end = len(targets) if args.end == -1 else args.end
    targets = set(targets[start: end])
    df = pd.read_csv(args.df)
    for idx, row in df.iterrows():
        fname, wo, ho = row['filename'], row['width'], row['height']
        if fname not in targets:
            continue
        if args.rtype == 'aspect':
            # keep aspect ratio
            rhigh, rlow, r = 1.0, 0.0, 1.0
            while abs(wo*r*ho*r - wt*ht) > thr:
                if wo*r*ho*r > wt*ht:
                    rhigh = r
                else:
                    rlow = r
                r = (rhigh + rlow)/2
            wti, hti = (int(wo*r)//round_size + 1)*round_size, (int(ho*r)//round_size + 1)*round_size
        elif args.rtype == 'naive':
            wti, hti = wt, ht
        else:
            raise AttributeError('Option for resizing type: naive|aspect')
        imdir_in = os.path.join(in_dir, fname)
        imnames = os.listdir(imdir_in)

        imdir_out = os.path.join(out_dir, fname)
        if not os.path.isdir(imdir_out):
            os.makedirs(imdir_out)
        # save image
        for imname in imnames:
            img = cv2.imread(os.path.join(imdir_in, imname))
            img_ = cv2.resize(img, (wti, hti))
            cv2.imwrite(os.path.join(imdir_out, imname), img_)
        if lambda_dir is not None:
            # save lambda
            for imname in imnames:
                lambda_idir = os.path.join(lambda_dir, fname)
                if not os.path.isdir(lambda_idir):
                    os.makedirs(lambda_idir)
                rec = {"rx": wo/wti, "ry": ho/hti, "dx": 0, "dy": 0}
                lambda_name = os.path.join(lambda_dir, fname, imname[:-3]+'json')
                json.dump(rec, open(lambda_name, "w"))
        print(fname)

if __name__ == '__main__':
    main()

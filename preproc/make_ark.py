from __future__ import print_function

import os
# import cv2
import time
import argparse
import pickle
import numpy as np
import skimage.io as skio
from collections import defaultdict

def make_ark(csv_file, input_dir, output_dir, start_id, end_id, partition, chunk_size=20, max_frame=1000, is_shuffle=True, ext_scp=None, as_gray=False):
    # read csv
    img_suffix = '.jpg'
    lns = open(csv_file, 'r').readlines()
    if not os.path.isdir(os.path.join(output_dir, partition)):
        print("Make dir: %s" % (output_dir))
        os.makedirs(os.path.join(output_dir, partition))
    scp_fn = os.path.join(output_dir, partition+'.scp')
    if ext_scp is not None:
        npz_to_imdir = defaultdict(list)
        imdirs = list(map(lambda x: x.strip().split(',')[0], lns))
        Ns = list(map(lambda x: int(x.strip().split(',')[2]), lns))
        imdir_to_N = dict(zip(imdirs, Ns))
        scp_rec = list(map(lambda x: x.strip().split(','), open(ext_scp, 'r').readlines()))
        for imdir, npz in scp_rec:
            npz_to_imdir[npz].append((imdir, imdir_to_N[imdir]))
    else:
        if is_shuffle:
            print('Shuffle subdirs')
            np.random.shuffle(lns)
        imdirs = list(map(lambda x: x.strip().split(',')[0], lns))
        Ns = list(map(lambda x: int(x.strip().split(',')[2]), lns))
        perm = range(len(imdirs))
        # imdict = {}
        idx, total_frames = 0, 0
        # fidx = str(args.start) + '_' + str(idx)
        scp_rec = []
        imdir_to_npz, npz_to_imdir = {}, defaultdict(list)
        if end_id == -1:
            end_id = len(perm)
        for i, pid in enumerate(perm):
            if not (i >= start_id and i < end_id):
                continue
            npz = os.path.join(partition + '/' + str(idx)+'.npz')
            total_frames += Ns[pid]
            imdir_to_npz[imdirs[pid]] = npz
            npz_to_imdir[npz].append((imdirs[pid], Ns[pid]))
            scp_rec.append((imdirs[pid], partition + '/' + str(idx)+'.npz'))
            if len(npz_to_imdir[npz]) == chunk_size or total_frames > max_frame:
                idx += 1
                total_frames = 0
    with open(scp_fn, 'w') as fo:
        print('Writing scp %s' % (scp_fn))
        for rec in scp_rec:
            fo.write(rec[0]+","+rec[1]+"\n")
    for npz, imdirs in npz_to_imdir.items():
        imdict = {}
        for imdir, N in imdirs:
            imnames = ["0"*(4-len(str(i))) + str(i) + img_suffix for i in range(1, N+1)]
            imgs = []
            for imname in imnames:
                img = skio.imread(os.path.join(input_dir, imdir, imname), as_gray=as_gray)
                print(img.shape)
                raise
                imgs.append(img)
            imgs = np.stack(imgs)
            imdict[imdir] = imgs
        np.savez_compressed(os.path.join(output_dir, npz), **imdict)
        print("%d items into %s" % (len(imdict), npz))
    return 0


def main():
    # source /share/data/asl-data/data/software/virtual_envs/pytorch1.1.0.cuda10/bin/activate
    #  python -B make_ark.py --input /share/data/asl-data/data/current_data/frames/IMG_ORG/AUG/FACEPATCH/RGB_4/ --output /share/data/asl-data/data/current_data/frames/IMG_ORG/AUG/FACEPATCH/RGB_4_npz/train/ --csv /share/data/asl-data/data/current_data/csv_ref/inhouse/train.csv --start 0 --end 6000 --chunk_size 100 --max_frame 3000 --partition train --shuffle
    parser = argparse.ArgumentParser(description='Make arks')
    parser.add_argument("--image", type=str, help="image dir")
    parser.add_argument("--ark", type=str, help="ark dir")
    parser.add_argument("--shuffle", action='store_true', help="shuffle list")
    parser.add_argument("--gray", action='store_true', help="gray-scale")
    parser.add_argument("--chunk_size", type=int, help="max number of items")
    parser.add_argument("--max_frame", type=int, default=1000, help="max number of frames")
    parser.add_argument("--csv", type=str, help="csv file")
    parser.add_argument("--partition", type=str, help="option: (train|dev|test)")
    parser.add_argument("--scp", type=str, default=None, help="scp file")
    parser.add_argument("--start", type=int, help="start id")
    parser.add_argument("--end", type=int, help="end id")
    args = parser.parse_args()
    make_ark(args.csv, args.image, args.ark, args.start, args.end, partition=args.partition, chunk_size=args.chunk_size, max_frame=args.max_frame, ext_scp=args.scp, as_gray=args.gray)

if __name__ == "__main__":
    main()

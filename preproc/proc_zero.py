## processing sequences with no face detected

from __future__ import print_function, division

import os
import cv2
import argparse
import numpy as np
import scipy.io as sio
from collections import defaultdict

parser = argparse.ArgumentParser(description="Processing sequences w.o face")
parser.add_argument("--rgb", type=str, help="image dir")
parser.add_argument("--bbox", type=str, help="face mat dir")
parser.add_argument("--csv", type=str, help="all sequences")
parser.add_argument("--zero", type=str, help="sequences with no face")
args = parser.parse_args()


def get_avg_face_size(img_dir, mat_dir, subdirs, w, h):
    N = 0
    avg_face = np.zeros((4), dtype=np.float32)
    for subdir in subdirs:
        imnames = os.listdir(os.path.join(img_dir, subdir))
        img = cv2.imread(os.path.join(img_dir, subdir, imnames[0]))
        if img.shape[0] != h or img.shape[1] != w:
            continue
        facemat = os.path.join(mat_dir, subdir, "face.mat")
        if os.path.isfile(facemat):
            avg_face = avg_face + sio.loadmat(facemat)["face"][0]
            N += 1
    avg_face = avg_face/N
    print(avg_face)
    return avg_face


def assign_face(img_dir, mat_dir, whole_csv, zero_fs):
    zero_subdirs = list(map(lambda x: x.strip().split(",")[0], open(zero_fs, "r").readlines()))
    whole_subdirs = list(map(lambda x: x.strip().split(",")[0], open(whole_csv, "r").readlines()))
    nonzero_subdirs = []
    # 1. average face of similar videos
    for subdir in whole_subdirs:
        if subdir not in zero_subdirs:
            nonzero_subdirs.append(subdir)
    video2subdir = defaultdict(list)
    for subdir in nonzero_subdirs:
        video_name = '_'.join(subdir.split('_')[:-3])
        video2subdir[video_name].append(subdir)
    zero_subdirs_ = []
    for subdir in zero_subdirs:
        video_name = '_'.join(subdir.split('_')[:-3])
        similar_subdirs = video2subdir[video_name]
        if len(similar_subdirs) != 0:
            avg_face = []
            for similar_subdir in similar_subdirs:
                facemat = os.path.join(mat_dir, similar_subdir, "face.mat")
                if os.path.isfile(facemat):
                    avg_face.append(sio.loadmat(facemat)["face"][0])
            avg_face = np.stack(avg_face).mean(axis=0).astype(np.int32)
            dir_to_save = os.path.join(mat_dir, subdir)
            if not os.path.isdir(dir_to_save):
                os.makedirs(dir_to_save)
            sio.savemat(os.path.join(dir_to_save, "face.mat"), {"face": avg_face})
        else:
            zero_subdirs_.append(subdir)
    zero_subdirs = zero_subdirs_
    # 2. average face of same size
    size2subdirs = defaultdict(list)
    for subdir in zero_subdirs:
        imnames = os.listdir(os.path.join(img_dir, subdir))
        img = cv2.imread(os.path.join(img_dir, subdir, imnames[0]))
        h, w = img.shape[0], img.shape[1]
        size2subdirs[(h, w)].append(subdir)
    for sz, imdirs in size2subdirs.items():
        h, w = sz[0], sz[1]
        avg_face_box = get_avg_face_size(img_dir, mat_dir, nonzero_subdirs, w, h)
        for imdir in imdirs:
            dir_to_save = os.path.join(mat_dir, imdir)
            if not os.path.isdir(dir_to_save):
                os.makedirs(dir_to_save)
            sio.savemat(os.path.join(dir_to_save, "face.mat"), {"face": avg_face_box})
    return 0


if __name__ == "__main__":
    assign_face(args.rgb, args.bbox, args.csv, args.zero)

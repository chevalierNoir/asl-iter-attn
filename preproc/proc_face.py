## preprocess face bbox

from __future__ import print_function
from __future__ import division

import os
import sys

import cv2
import json
import numpy as np
import scipy.io as sio
import argparse
import warnings

warnings.simplefilter('error')

def get_motionness(opt_file, boxes, ratios=(1.5, 1.5, 1, 1.5)):
    # only use opt value
    left_ratio, right_ratio, up_ratio, down_ratio = ratios[0], ratios[1], ratios[2], ratios[3]
    opt = cv2.imread(opt_file, 0)
    scores, hand_patches = [], []
    for i in range(len(boxes)):
        box = boxes[i]
        w, h = box[2]-box[0], box[3]-box[1]
        x0, y0, x1, y1 = int(box[0]-w*left_ratio), int(box[1]-h*up_ratio), int(box[2]+w*right_ratio), int(box[3]+h*down_ratio)
        left_expand = -x0 if x0 < 0 else 0
        up_expand = -y0 if y0 < 0 else 0
        right_expand = x1-opt.shape[1]+1 if x1 > opt.shape[1]-1 else 0
        down_expand = y1-opt.shape[0]+1 if y1 > opt.shape[0]-1 else 0
        expand_opt = cv2.copyMakeBorder(opt, up_expand, down_expand, left_expand, right_expand, cv2.BORDER_CONSTANT, (0, 0, 0))
        # normalization of pixel value
        score = (expand_opt[y0+up_expand: y1+up_expand, x0+left_expand: x1+left_expand]/255.0).mean()
        scores.append(score)
    return scores


def greedy(boxes, scores, thr_iu=0.5):
    def check_empty(boxes):
        T = len(boxes)
        empty = True
        for t in range(T):
            if len(boxes[t]) != 0:
                empty = False
                break
        return empty

    def get_iu(box, grt):
        # box, grt: [x1, y1, x2, y2]
        box, grt = list(map(float, box)), list(map(float, grt))
        inter_x = max(min(box[2], grt[2])-max(box[0], grt[0])+1, 0)
        inter_y = max(min(box[3], grt[3])-max(box[1], grt[1])+1, 0)
        inter_area = inter_x * inter_y
        union_area = (box[2]-box[0]+1)*(box[3]-box[1]+1)+(grt[2]-grt[0]+1)*(grt[3]-grt[1]+1)-inter_area
        iu = inter_area/union_area
        return iu

    T = len(boxes)
    tubes, score_tubes = [], []
    while not check_empty(boxes):
        tube, score_tube = [], []
        # init
        for t in range(T):
            if len(boxes[t]) != 0:
                st_fid = t
                st_box, st_score = boxes[t].pop(0), scores[t].pop(0)
                break
        tube.append(st_box)
        score_tube.append(st_score)
        # iter
        for t in range(st_fid, T):
            if len(boxes[t]) == 0:
                continue
            for i in range(len(boxes[t])):
                iu = get_iu(boxes[t][i], tube[-1])
                if iu > thr_iu:
                    box_t, score_t = boxes[t].pop(i), scores[t].pop(i)
                    tube.append(box_t)
                    score_tube.append(score_t)
                    break
        tubes.append(tube)
        score_tubes.append(sum(score_tube))
    idx = score_tubes.index(max(score_tubes))
    tube = tubes[idx]
    box_coords = [0, 0, 0, 0]
    for s in range(len(tube)):
        box_coords[0] = box_coords[0]+tube[s][0]
        box_coords[1] = box_coords[1]+tube[s][1]
        box_coords[2] = box_coords[2]+tube[s][2]
        box_coords[3] = box_coords[3]+tube[s][3]
    if len(tube) == 0:
        raise ValueError("No tube found")
    box_coords = list(map(lambda x: x/len(tube), box_coords))
    return box_coords


def find_signer_face(in_dir, out_dir, opt_dir, csv, log, ratio, start, end):

    def int2str(idx):
        return "0"*(4-len(str(idx)))+str(idx)
    with open(csv, "r") as fo:
        subdirs = list(map(lambda x: x.strip().split(",")[0], fo.readlines()))
    end = len(subdirs) if end == -1 else end
    log_zero = log
    zero_subdirs = []
    for i, subdir in enumerate(subdirs):
        if not (i >= start and i < end):
            continue
        matnames = sorted(os.listdir(os.path.join(in_dir, subdir)))
        if len(matnames) == 0:
            zero_subdirs.append(subdir)
            continue
        num_frames = len(os.listdir(os.path.join(opt_dir, subdir)))
        cand_face_boxes = [None for _ in range(num_frames)]
        for i in range(num_frames):
            matname = int2str(i+1)+".mat"
            if matname in matnames:
                matname_full = os.path.join(in_dir, subdir, matname)
                face_box = sio.loadmat(matname_full)["face"]
                if face_box.shape[1] == 6:
                    face_box = face_box[:, :-2]  # only x0, y0, x1, y1
                elif face_box.shape[1] == 4 or face_box.shape[1] == 0:
                    face_box = face_box
                else:
                    raise ValueError("2nd dimension: 4 or 6 or 0")
                cand_face_boxes[i] = face_box
            else:
                # if not found, use prev one
                cand_face_boxes[i] = cand_face_boxes[i-1].copy()
        face_boxes, face_scores = [], []
        for idx, matname in enumerate(matnames):
            opt_file = os.path.join(opt_dir, subdir, matname.split(".")[0]+".jpg")
            face_box = cand_face_boxes[idx]
            face_box_ = []
            for i in range(len(face_box)):
                aspect_ratio = (face_box[i, 2] - face_box[i, 0])/max((face_box[i, 3] - face_box[i, 1]), 1)
                if aspect_ratio > 0.5 and aspect_ratio < 2 and face_box[i].sum() > 0:
                    face_box_.append(face_box[i, :])
            if len(face_box_) == 0:
                continue
            face_box = np.stack(face_box_).tolist()
            face_score = get_motionness(opt_file, face_box, ratio)
            face_boxes.append(face_box)
            face_scores.append(face_score)
        if len(face_boxes) == 0:
            zero_subdirs.append(subdir)
            continue
        try:
            avg_face_box = greedy(face_boxes, face_scores)
        except ValueError:
            zero_subdirs.append(subdir)
            continue
        avg_face_box = np.array(avg_face_box, dtype=np.float32)
        out_fulldir = os.path.join(out_dir, subdir)
        if not os.path.isdir(out_fulldir):
            os.makedirs(out_fulldir)
        out_fullname = os.path.join(out_dir, subdir, "face.mat")
        sio.savemat(out_fullname, {"face": avg_face_box})
        print(subdir)
    with open(log_zero, 'w') as fo:
        fo.write('\n'.join(zero_subdirs))
    return 0


def rescale_frame(in_dir, out_dir, face_dir, lambda_dir, csv, start, end, base_size, max_square):
    # in_dir: input image dir, out_dir: output image dir, face_dir: face.mat dir, lambda_dir: output lambda
    # base_size: (h, w) of face box, max_square: maximum square of resized image
    with open(csv, "r") as fo:
        subdirs = map(lambda x: x.strip().split(",")[0], fo.readlines())
    bxy = base_size
    decay_rate = 0.95
    N, Ns = len(subdirs), 0
    for i, subdir in enumerate(subdirs):
        if not (i >= start and i < end):
            continue
        imnames = os.listdir(os.path.join(in_dir, subdir))
        img = cv2.imread(os.path.join(in_dir, subdir, imnames[0]))
        h, w = img.shape[0], img.shape[1]
        face_bbox = sio.loadmat(os.path.join(face_dir, subdir, "face.mat"))["face"][0]
        axy = max(face_bbox[2]-face_bbox[0], face_bbox[3]-face_bbox[1])
        alpha = 1
        nxy = bxy/axy
        while ((w*nxy)//32 + 1)*((h*nxy)//32 + 1)*32*32 > max_square:
            alpha = alpha*decay_rate
            nxy = nxy*alpha
        if alpha < 1:
            Ns += 1
            print(subdir, Ns)
        ws, hs = int(w*nxy), int(h*nxy)
        img_save_dir, lambda_save_dir = os.path.join(out_dir, subdir), os.path.join(lambda_dir, subdir)
        if not os.path.isdir(img_save_dir):
            os.makedirs(img_save_dir)
        if not os.path.isdir(lambda_save_dir):
            os.makedirs(lambda_save_dir)
        for i, imname in enumerate(imnames):
            imname_to_save = imname.split(".")[0]+".jpg"
            img = cv2.imread(os.path.join(in_dir, subdir, imname))
            # resize and pad
            img_to_save = cv2.resize(img, (ws, hs))
            xpad, ypad = int(((w*nxy)//32 + 1)*32)-ws, int(((h*nxy)//32 + 1)*32)-hs
            img_to_save = cv2.copyMakeBorder(img_to_save, 0, ypad, 0, xpad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            if img_to_save.shape[0] % 32 != 0 or img_to_save.shape[1] % 32 != 0:
                raise ValueError("size not multiples of 32", subdir)
            cv2.imwrite(os.path.join(img_save_dir, imname_to_save), img_to_save)
            lambda_name = os.path.join(lambda_save_dir, imname.split(".")[0]+".json")
            rec = {"rx": 1/nxy, "ry": 1/nxy, "dx": 0, "dy": 0, "scale": alpha}  # 1/rx
            with open(lambda_name, "w") as fo:
                json.dump(rec, fo)
        print(subdir)
    print("ratio of face smaller than threshold: %.3f" % (100*Ns/N))
    return 0


def crop_frame(in_dir, out_dir, face_dir, lambda_dir, csv, start, end, ratios, imsize=224):
    # in_dir: input image dir, out_dir: output image dir, face_dir: face.mat dir, lambda_dir: output lambda
    left_ratio, right_ratio, up_ratio, down_ratio = ratios[0], ratios[1], ratios[2], ratios[3]
    with open(csv, "r") as fo:
        subdirs = list(map(lambda x: x.strip().split(",")[0], fo.readlines()))
    end = len(subdirs) if end == -1 else end
    for i, subdir in enumerate(subdirs):
        if not (i >= start and i < end):
            continue
        box = sio.loadmat(os.path.join(face_dir, subdir, "face.mat"))["face"][0]
        wbox, hbox = box[2]-box[0], box[3]-box[1]
        x0, y0, x1, y1 = int(box[0]-wbox*left_ratio), int(box[1]-hbox*up_ratio), int(box[2]+wbox*right_ratio), int(box[3]+hbox*down_ratio)
        imnames = os.listdir(os.path.join(in_dir, subdir))
        img_save_dir, lambda_save_dir = os.path.join(out_dir, subdir), os.path.join(lambda_dir, subdir)
        if not os.path.isdir(img_save_dir):
            os.makedirs(img_save_dir)
        if not os.path.isdir(lambda_save_dir):
            os.makedirs(lambda_save_dir)
        for imname in imnames:
            img = cv2.imread(os.path.join(in_dir, subdir, imname))
            left_expand_img = -x0 if x0 < 0 else 0
            up_expand_img = -y0 if y0 < 0 else 0
            right_expand_img = x1-img.shape[1]+1 if x1 > img.shape[1]-1 else 0
            down_expand_img = y1-img.shape[0]+1 if y1 > img.shape[0]-1 else 0
            expand_img = cv2.copyMakeBorder(img, up_expand_img, down_expand_img, left_expand_img, right_expand_img, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            patch = expand_img[y0+up_expand_img: y1+up_expand_img, x0+left_expand_img: x1+left_expand_img]
            up_expand_patch, down_expand_patch, left_expand_patch, right_expand_patch = 0, 0, 0, 0
            if patch.shape[0] < patch.shape[1]:
                up_expand_patch, down_expand_patch = (patch.shape[1]-patch.shape[0])//2, (patch.shape[1]-patch.shape[0])//2 + 1
            elif patch.shape[0] > patch.shape[1]:
                left_expand_patch, right_expand_patch = (patch.shape[0]-patch.shape[1])//2, (patch.shape[0]-patch.shape[1])//2 + 1
            square_patch = cv2.copyMakeBorder(patch, up_expand_patch, down_expand_patch, left_expand_patch, right_expand_patch, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            rx, ry = square_patch.shape[1]/imsize, square_patch.shape[0]/imsize
            dx, dy = -x0+left_expand_patch, -y0+up_expand_patch
            square_patch = cv2.resize(square_patch, (imsize, imsize))
            imname_to_save = os.path.join(img_save_dir, imname.split(".")[0]+".jpg")
            cv2.imwrite(imname_to_save, square_patch)
            lambda_name = os.path.join(lambda_save_dir, imname.split(".")[0]+".json")
            rec = {"rx": rx, "ry": ry, "dx": dx, "dy": dy}
            with open(lambda_name, "w") as fo:
                json.dump(rec, fo)
        print(subdir)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocessing")
    parser.add_argument("--task", type=str, help="task: face|rescale|crop")
    parser.add_argument("--input_dir", type=str, help="input dir")
    parser.add_argument("--output_dir", type=str, help="output dir")
    parser.add_argument("--opt_dir", type=str, help="opt dir")
    parser.add_argument("--log_dir", type=str, help="log dir")
    parser.add_argument("--lambda_dir", type=str, help="lambda dir")
    parser.add_argument("--face_dir", type=str, help="facemat dir")
    parser.add_argument("--csv", type=str, help="csv file")
    parser.add_argument("--zero", type=str, help="sequences with no face")
    parser.add_argument("--start", type=int, help="start index")
    parser.add_argument("--end", type=int, help="end index")
    args = parser.parse_args()
    if args.task == "face":
        find_signer_face(args.input_dir, args.output_dir, args.opt_dir, args.csv, args.log_dir, ratio=(1.5, 1.5, 1.5, 1.5), start=args.start, end=args.end)
    elif args.task == "rescale":
        rescale_frame(args.input_dir, args.output_dir, args.face_dir, args.lambda_dir, args.csv, start=args.start, end=args.end, base_size=36, max_square=224*224)
    elif args.task == "crop":
        crop_frame(args.input_dir, args.output_dir, args.face_dir, args.lambda_dir, args.csv, start=args.start, end=args.end, ratios=(1.5, 1.5, 1.5, 1.5), imsize=224)
    else:
        raise ValueError("Option for task: face|rescale|crop")

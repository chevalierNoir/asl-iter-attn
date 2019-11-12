import os
import cv2
import vtb
import json
import argparse
import numpy as np
import scipy.io as sio


def idx2name(idx):
    return str(idx).zfill(4)


def aug_resolution(subdir, in_dir, out_dir, beta_dir, lambda_dirs, scale, hw):
    h, w = hw[0], hw[1]
    lambda_cur, lambda_dirs = lambda_dirs[-1], lambda_dirs[:-1][::-1]
    in_dir_full, out_dir_full = os.path.join(in_dir, subdir), os.path.join(out_dir, subdir)
    imnames = sorted(os.listdir(in_dir_full))
    if not os.path.isdir(out_dir_full):
        os.makedirs(out_dir_full)
    for i in range(len(imnames)):
        inname_full, outname_full = os.path.join(in_dir, subdir, imnames[i]), os.path.join(out_dir, subdir, imnames[i])
        img = cv2.imread(inname_full)
        s = json.load(open(os.path.join(lambda_cur, subdir, idx2name(i+1)+'.json'), 'r'))
        cx, cy = s['cx'], s['cy']
        x0, y0, x1, y1 = cx-scale*w/2, cy-scale*h/2, cx+scale*w/2, cy+scale*h/2
        for j in range(len(lambda_dirs)):
            lambda_name_full = os.path.join(lambda_dirs[j], subdir, idx2name(i+1)+".json")
            with open(lambda_name_full, "r") as fo:
                s = json.load(fo)
            rx, ry, dx, dy = s["rx"], s["ry"], s["dx"], s["dy"]
            if "scale" in s:
                mx, my, sw, sh = (x0+x1)/2.0, (y0+y1)/2.0, s["scale"]*(x1-x0+1), s["scale"]*(y1-y0+1)
                x0, x1, y0, y1 = mx - sw/2, mx + sw/2, my - sh/2, my + sh/2
            x0, y0, x1, y1 = x0*rx-dx, y0*ry-dy, x1*rx-dx, y1*ry-dy
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        left_expand = -x0 if x0 < 0 else 0
        up_expand = -y0 if y0 < 0 else 0
        right_expand = x1-img.shape[1]+1 if x1 > img.shape[1]-1 else 0
        down_expand = y1-img.shape[0]+1 if y1 > img.shape[0]-1 else 0
        expand_img = cv2.copyMakeBorder(img, up_expand, down_expand, left_expand, right_expand, cv2.BORDER_CONSTANT, (0, 0, 0))
        hand_patch = expand_img[y0+up_expand: y1+up_expand, x0+left_expand: x1+left_expand]
        hand_patch = cv2.resize(hand_patch, (h, w))
        cv2.imwrite(outname_full, hand_patch)
    return 0


def search(beta_name_full, hw, avg=1, mode="vtb"):
    h, w = hw[0], hw[1]
    beta = sio.loadmat(beta_name_full)["beta"]
    L, rows, cols = beta.shape[0], beta.shape[1], beta.shape[2]
    beta_flatten = beta.reshape(L, rows*cols)
    if mode == "vtb":
        num_points, scale, sr, smoothing = 2, 0.8, 0.8, 0.01  # sr: range to compute score
        boxes, scores = [[] for _ in range(L)], [[] for _ in range(L)]
        max_idx = np.argsort(beta_flatten, axis=1)[:, -num_points:]
        for i in range(L):
            for j in range(num_points):
                x, y = max_idx[i][j] % cols, max_idx[i][j]/cols
                x0, y0, x1, y1 = x-cols*scale/2, y-rows*scale/2, x+cols*scale/2, y+rows*scale/2
                xs0, ys0, xs1, ys1 = x-cols*sr/2, y-rows*sr/2, x+cols*sr/2, y+rows*sr/2
                score = beta[i][int(max(ys0, 0)): int(ys1), int(max(xs0, 0)): int(xs1)].sum()/((xs1-xs0)*(ys1-ys0))
                boxes[i].append([x0, y0, x1, y1])
                scores[i].append(score)
        vtb_xy = vtb.vtb(boxes, scores, smoothing)
        max_xys = np.array(list(map(lambda x: [(x[0]+x[2])/2, (x[1]+x[3])/2], vtb_xy)), dtype=np.int32)
    else:
        max_idx = np.argmax(beta_flatten, axis=1)
        max_xys = np.stack((max_idx % cols, max_idx/cols), axis=1)
    rec_ratio_x, rec_ratio_y = w/float(cols), h/float(rows)
    cxs, cys = [], []
    for i in range(L):
        if avg == 1:
            cx, cy = int(np.mean(max_xys, axis=0)[0]*rec_ratio_x), int(np.mean(max_xys, axis=0)[1]*rec_ratio_y)
        else:
            cx, cy = int(max_xys[i][0]*rec_ratio_x), int(max_xys[i][1]*rec_ratio_y)
        cxs.append(cx)
        cys.append(cy)
    return cxs, cys


def get_lambda(subdir, beta_dir, lambda_dir, scale, roi_hw=(224, 224), avg=1):
    # get coef to retrieve patch: left, right, rx, ry
    # roi_hw: h/w of roi
    lambda_subdir_full = os.path.join(lambda_dir, subdir)
    if not os.path.isdir(lambda_subdir_full):
        os.makedirs(lambda_subdir_full)
    beta_name_full = os.path.join(beta_dir, subdir, "beta.mat")
    hw = sio.loadmat(beta_name_full)["hw"][0]
    cxs, cys = search(beta_name_full, hw, avg)
    for i in range(len(cxs)):
        cx, cy = cxs[i], cys[i]
        x0, y0 = int(cx-scale*roi_hw[1]/2), int(cy-scale*roi_hw[0]/2)
        rx, ry = scale, scale
        dx, dy = -x0, -y0
        lambda_name = os.path.join(lambda_dir, subdir, idx2name(i+1)+".json")
        rec = {"rx": rx, "ry": ry, "dx": dx, "dy": dy, "cx": cx, "cy": cy}
        json.dump(rec, open(lambda_name, "w"))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aug resolution")
    parser.add_argument("--input", type=str, help="input image dir")
    parser.add_argument("--output", type=str, help="output image dir")
    parser.add_argument("--csv", type=str, help="csv file")
    parser.add_argument("--beta", type=str, help="beta dir")
    parser.add_argument("--lambda_", type=str, nargs='+', help="lambda dir")
    parser.add_argument("--scale", type=float, help="scaling ratio")
    parser.add_argument("--hw", type=int, nargs="+", help="height/width of cropped image")
    parser.add_argument("--avg", type=int, default=1, help="avg bbox when computing lambda")
    parser.add_argument("--save_lambda", action='store_true', help="save lambda dir")
    parser.add_argument("--save_img", action='store_true', help="save image")
    parser.add_argument("--start", type=int, help="start id")
    parser.add_argument("--end", type=int, help="end id")
    args = parser.parse_args()
    in_dir, out_dir, beta_dir, lambdas, scale, hw = args.input, args.output, args.beta, args.lambda_, args.scale, args.hw
    subdirs = []
    with open(args.csv, "r") as fo:
        lns = fo.readlines()
        for i in range(len(lns)):
            imdir = lns[i].strip().split(",")[0]
            subdirs.append(imdir)
    start = args.start
    if args.end < -1:
        raise ValueError('end >= -1')
    end = len(subdirs) if args.end == -1 else args.end
    if args.save_lambda:
        print("avg: %d" % (args.avg))
        for i, subdir in enumerate(subdirs):
            if not (i >= start and i < end):
                continue
            get_lambda(subdir, beta_dir, lambdas[0], scale, hw, avg=args.avg)
    if args.save_img:
        print('Iter-zoom on', in_dir)
        for i, subdir in enumerate(subdirs):
            if not (i >= start and i < end):
                continue
            aug_resolution(subdir, in_dir, out_dir, beta_dir, lambdas, scale, hw)

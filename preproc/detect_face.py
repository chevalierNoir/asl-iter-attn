import os
import cv2
import face_recognition
import numpy as np
import scipy.io as sio
import argparse
from collections import defaultdict, Counter

def get_face(img_file, mat_file):
    try:
        img = face_recognition.load_image_file(img_file)
        locations = face_recognition.face_locations(img, model='cnn') # , model="cnn"
    except Exception:
        return -1
    boxes = []
    for i in range(len(locations)):
        top, right, bottom, left = locations[i]
        boxes.append([left, top, right, bottom])
    boxes = np.array(boxes, dtype=np.int32)
    sio.savemat(mat_file, {"face": boxes})
    return len(boxes)


def save_bbox_face(read_dir, write_dir, csv, start, end, min_samples):
    with open(csv, "r") as fo:
        subdirs = list(map(lambda x: x.strip().split(",")[0], fo.readlines()))
    end = end if end != -1 else len(subdirs)
    zero_subdirs = []
    for i, subdir in enumerate(subdirs):
        if not (i >= start and i < end):
            continue
        read_dir_full, write_dir_full = os.path.join(read_dir, subdir), os.path.join(write_dir, subdir)
        if not os.path.isdir(write_dir_full):
            os.makedirs(write_dir_full)
        imnames = sorted(os.listdir(os.path.join(read_dir, subdir)))
        imnames_to_select = imnames[::min_samples]
        for imname in imnames:
            if len(imnames_to_select) >= min_samples:
                break
            if imname not in imnames_to_select:
                imnames_to_select.append(imname)
        status = True
        for imname in imnames_to_select:
            read_imname = os.path.join(read_dir_full, imname)
            write_matname = os.path.join(write_dir_full, imname.split(".")[0]+".mat")
            ni = get_face(read_imname, write_matname)
            if ni == -1:
                status = False
                break
        print(subdir)
        if not status:
            zero_subdirs.append(subdir)
    return


def main():
    parser = argparse.ArgumentParser(description="run facerecognizer")
    parser.add_argument("--rgb", type=str, help="image dir")
    parser.add_argument("--bbox", type=str, help="bbox dir")
    parser.add_argument("--csv", type=str, help="csv file")
    parser.add_argument("--start", type=int, help="start id")
    parser.add_argument("--end", type=int, help="end id")
    args = parser.parse_args()
    save_bbox_face(args.rgb, args.bbox, args.csv, args.start, args.end, min_samples=3)
    return

if __name__ == "__main__":
    main()

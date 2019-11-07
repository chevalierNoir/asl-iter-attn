import os
import argparse
import pandas as pd
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='make csv files for model training')
    parser.add_argument('--iraw', type=str, help='raw csv file')
    parser.add_argument('--odir', type=str, help='dir of processed files')
    parser.add_argument('--ofn', type=str, help='file of all partitions')
    args = parser.parse_args()
    df = pd.read_csv(args.iraw)
    csv_dir = args.odir
    if not os.path.isdir(csv_dir):
        print('Make dir', csv_dir)
        os.makedirs(csv_dir)
    pdata = defaultdict(list)
    for _, row in df.iterrows():
        pdata[row['partition']].append([row['filename'], row['label_proc'], str(row['number_of_frames'])])
    for part, lns in pdata.items():
        fn = os.path.join(csv_dir, part+'.csv')
        with open(fn, 'w') as fo:
            for ln in lns:
                fo.write(','.join(ln)+'\n')
    whole_csv = args.ofn
    whole_dir = '/'.join(whole_csv.split('/')[:-1])
    if not os.path.isdir(whole_dir):
        print('Make dir ', whole_dir)
        os.makedirs(whole_dir)
    with open(whole_csv, 'w') as fo:
        for part, lns in pdata.items():
            for ln in lns:
                fo.write(','.join(ln)+'\n')
    return

if __name__ == '__main__':
    main()

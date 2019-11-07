data_dir=~/data

# dir of raw frames (From data tarball)
rgb_dir=$data_dir/RGB

# original csv file (From data tarball)
pd_csv=$data_dir/ChicagoFSWild.csv

# dir containing processed csv files
csv_dir=$data_dir/total-csv

# csv file of all partitions (train+val+test)
whole_csv=$csv_dir/whole.csv

 # ImageNet pre-trained AlexNet (Download: https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth)
prt_conv=$data_dir/alexnet-owt-4df8aa71.pth

# opt-flow dir
opt_dir=$data_dir/OPT

# prior dir
prior_dir=$data_dir/PRIOR

# dir containing intermediate results from iterative attention
iter_dir=$data_dir/iter

# language model dir (For decoding)
lm_dir=$data_dir/lm

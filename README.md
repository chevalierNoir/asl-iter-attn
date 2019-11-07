# ASL Fingerspelling recognition in the wild

This is the code for paper:
- **Fingerspelling recognition in the wild with iterative visual attention** (Shi et al. ICCV 2019)[[paper](https://arxiv.org/pdf/1908.10546.pdf)] 


## Preparation:
1. Download [data](https://ttic.uchicago.edu/~klivescu/ChicagoFSWild.htm#download)

2. Major dependencies:
python 2.7+ (Should be compatible with 3.x)
pytorch 0.4.1
OpenCV 3+
[Warp-ctc](https://github.com/SeanNaren/warp-ctc)
[face_recognition](https://github.com/ageitgey/face_recognition) (required for Face-ROI setting)

## Experiment pipeline:
1. Untar ChicagoFSWild.tgz and edit paths in path.sh

2. Preprocessing:
```sh
cd preproc
./preproc.sh
```
Pre-processing mainly includes (1). Getting optical flow of frame sequence (2). Generating prior for each frame (3). Resizing image frame (4). Detecting face and cropping initial ROI for model training (Face-ROI only)

Switch `task` variable between `face-roi` and `whole` for preprocessing for Face-ROI setting and Whole-frame setting.
Note: if you are on cluster, you may accelerate this process by launching multiple jobs simultaneously through specifying `start` and `end` variable which denotes the start/end sequence id.

3. Model training:

Whole-frame experiment:
```sh
./whole_exp.sh
```
Face-ROI experiment:
```sh
./faceroi_exp.sh
```

You may specify different hyperparameters in `conf.ini`. Note current data loader does not support having frame sequences of different width/height within one batch. All images are resized to same size beforehand except for training in the first iteration under whole-frame setting. Batch size for that iteration must be set to 1.

4. Decoding and Evaluation:

Quick run: set `stage` variable to 2 in `whole_exp.sh` or `faceroi_exp.sh` then re-run two scripts. This will do (1). generating probability sequence for each frame sequence (2). greedy decoding (3). training language model (4). beam search with language model.

You can also do these steps separately, especially for tuning parameters in (4) including beam width, language model weights and insertion/deletion penalty for a given language model. 

import os
import torch
import scipy.io as sio
import skimage.io as skio
import numpy as np
import torch.utils.data as tud
from torch.nn import functional as F

class SLData(tud.Dataset):
    """
    Sign Language Dataset
    """
    def __init__(self, img_dir, prior_dir, fcsv, vocab_map, transform=None, upper_len=200, upper_sample=2):
        # upper_len: frame sub-sample if length larger
        self.img_dir = img_dir
        self.prior_dir = prior_dir
        self.fcsv = fcsv
        self.vocab_map = vocab_map
        self.transform = transform
        self.upper_len = upper_len
        self.upper_sample = upper_sample
        self._parse()

    def _parse(self):
        with open(self.fcsv, "r") as fo:
            lns = fo.readlines()
        # sub-sampling
        print("%d data" % len(lns))
        self.imdirs, self.labels, self.num_frames = [], [], []
        for i in range(len(lns)):
            imdir, label, nframe = lns[i].strip().split(",")
            self.imdirs.append(imdir)
            self.labels.append(label)
            self.num_frames.append(int(nframe))

    def __len__(self):
        return len(self.imdirs)

    def _int2str(self, i):
        return "0"*(4-len(str(i))) + str(i)

    def __getitem__(self, idx):
        label = map(lambda x: self.vocab_map[x], self.labels[idx])
        fnames = [self._int2str(i)+".jpg" for i in range(1, self.num_frames[idx]+1)]
        imgs, priors = [], []
        for fname in fnames:
            im_fullname = os.path.join(self.img_dir, self.imdirs[idx], fname)
            img = skio.imread(im_fullname).astype(np.float32)
            imgs.append(img)
        imgs = np.stack(imgs)
        prior_fname = os.path.join(self.prior_dir, self.imdirs[idx], 'prior.npy')
        prior = np.load(prior_fname)
        if len(imgs) > self.upper_len:
            imgs, prior = imgs[::self.upper_sample], prior[::self.upper_sample]

        sample = {'image': imgs, 'prior': prior, 'label': label, 'prob_size': [len(imgs)], 'label_size': [len(label)], 'imdir': self.imdirs[idx]}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

class Rescale(object):
    """Scale Image/Prior tensor (data augmentation)"""
    def __init__(self, scales, hws, origin_scale=False):
        self.scales = sorted(scales)
        self.hs = sorted([hw[0] for hw in hws])
        self.ws = sorted([hw[1] for hw in hws])
        self.origin = origin_scale

    def __call__(self, sample):
        if self.origin:
            sample['image'] = sample['image'].unsqueeze(dim=0)
            sample['prior'] = sample['prior'].unsqueeze(dim=0)
            return sample

        H, W = sample['image'].size(2), sample['image'].size(3)
        Hmax, Wmax = int(H*self.scales[-1]), int(W*self.scales[-1])
        hmax, wmax = self.hs[-1], self.ws[-1]
        images, priors = [], []
        with torch.no_grad():
            for i, scaling in enumerate(self.scales):
                X = F.upsample(sample['image'], size=(int(H*scaling), int(W*scaling)), mode='bilinear')
                M = F.upsample(sample['prior'].unsqueeze(dim=1), size=(self.hs[i], self.ws[i]), mode='bilinear')
                Xmax = X.new_zeros((len(sample['image']), 3, Hmax, Wmax))
                Mmax = M.new_zeros((len(sample['prior']), 1, hmax, wmax))
                x0, y0, x1, y1 = max((Wmax - X.size(-1))//2, 0), max((Hmax - X.size(-2))//2, 0), min((Wmax + X.size(-1))//2, Wmax), min((Hmax + X.size(-2))//2, Hmax)
                Xmax[:, :, y0: y1, x0: x1] = X
                x0, y0, x1, y1 = max((wmax - M.size(-1))//2, 0), max((hmax - M.size(-2))//2, 0), min((wmax + M.size(-1))//2, wmax), min((hmax + M.size(-2))//2, hmax)
                Mmax[:, :, y0: y1, x0: x1] = M
                images.append(Xmax)
                priors.append(Mmax)
        sample['image'], sample['prior'] = torch.stack(images), torch.stack(priors).squeeze(dim=2)
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        image, prior, label, prob_size, label_size = sample['image'], sample['prior'], sample['label'], sample['prob_size'], sample['label_size']
        # swap color axis because
        # numpy image: L x H x W x C
        # torch image: L x C X H X W
        image, prior, label = torch.from_numpy(image).to(self.device), torch.from_numpy(prior).to(self.device), torch.IntTensor(label)
        prob_size, label_size = torch.IntTensor(prob_size), torch.IntTensor(label_size)
        image = image.transpose(2, 3).transpose(1, 2)
        sample = {'image': image, 'prior': prior, 'label': label, 'prob_size': prob_size, 'label_size': label_size, 'imdir': sample['imdir']}
        return sample

class Normalize(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, mean, std, device):
        self.mean = torch.FloatTensor(mean).to(device).view(1, 1, 3, 1, 1)
        self.std = torch.FloatTensor(std).to(device).view(1, 1, 3, 1, 1)

    def __call__(self, sample):
        image = sample['image']
        image = (image/255.0 - self.mean) / self.std
        sample['image'] = image
        return sample

class Pad(object):
    """Pad tensors"""
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, sample):
        if len(sample['image']) < self.max_len:
            n = self.max_len - len(sample['image'])
            sz = [n] + list(sample['image'].size()[1:])
            padded = sample['image'].new_zeros(*sz)
            sample['image'] = torch.cat((sample['image'], padded), dim=0)
        return sample

def collate_fn_ctc(data):
    num_scales = data[0]['image'].size(0)
    bsz, Nmax, imsz = len(data)*data[0]['image'].size(0), max([x['image'].size(1) for x in data]), list(data[0]['image'].size()[2:])
    psz = list(data[0]['prior'].size()[2:])
    with torch.no_grad():
        frames, priors = data[0]['image'].new_zeros(*([bsz, Nmax] + imsz)), data[0]['prior'].new_zeros(*([bsz, Nmax] + psz))
        labels, prob_sizes, label_sizes = [], [], []
        for i in range(len(data)):
            frames[i*num_scales: (i+1)*num_scales, :data[i]['image'].size(1)] = data[i]['image']
            priors[i*num_scales: (i+1)*num_scales, :data[i]['prior'].size(1)] = data[i]['prior']
            labels.extend([data[i]['label'] for _ in range(num_scales)])
            prob_sizes.extend([data[i]['prob_size'] for _ in range(num_scales)])
            label_sizes.extend([data[i]['label_size'] for _ in range(num_scales)])
        labels, prob_sizes, label_sizes = torch.cat(labels), torch.cat(prob_sizes), torch.cat(label_sizes)
    imdir = [d['imdir'] for d in data for _ in range(num_scales)]
    sample = {'image': frames, 'prior': priors, 'label': labels, 'prob_size': prob_sizes, 'label_size': label_sizes, 'imdir': imdir}
    return sample

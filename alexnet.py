import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, output_size, p_2d=0.25):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.Dropout2d(p=p_2d),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.Dropout2d(p=p_2d),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Dropout2d(p=p_2d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        return x


def alexnet(pretrain=None, **kwargs):
    model = AlexNet(**kwargs)
    if pretrain is not None:
        print("Load pre-trained model: %s" % (pretrain))
        uninit_keys = ['classifier.1.weight', 'classifier.1.bias', 'classifier.4.weight', 'classifier.4.bias', 'classifier.6.weight', 'classifier.6.bias']
        mismatch_keys = {"features.8.weight": "features.9.weight",
                         "features.8.bias": "features.9.bias",
                         "features.10.weight": "features.12.weight",
                         "features.10.bias": "features.12.bias"}
        std = torch.load(pretrain)
        for pname, param in std.items():
            if pname in uninit_keys:
                continue
            elif pname in mismatch_keys:
                model.state_dict()[mismatch_keys[pname]].copy_(param)
            else:
                model.state_dict()[pname].copy_(param)
    return model

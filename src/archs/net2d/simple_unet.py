import torch
from torch import nn

from src.archs.net2d.base import _act_layer, _norm_layer
from src.archs.net2d.unet import _up_layer

cfg_template = {
    'encoder': [[64], ["max", 128], ["max", 256, 256], ["max", 512, 512], ],
    'mid': ["max", 512, 512],
    'decoder': [[512, 512], [256, 256], [128, ], [64, ], ]
}


class UNet(nn.Module):

    def __init__(self, in_chs, out_chs, cfg, use_norm=False, init_weights=True, **kwargs):
        super().__init__()
        self.use_norm = use_norm
        self.kwargs = kwargs
        up_mode = kwargs.pop('up_mode') if 'up_mode' in kwargs else 'tconv'

        current_chs = in_chs

        encoders = []
        skip_chs = []
        for e in cfg["encoder"]:
            encoder, current_chs = self.make_layer(current_chs, e)
            encoders.append(encoder)
            skip_chs.insert(0, current_chs)
        self.encoders = nn.ModuleList(encoders)

        mid, current_chs = self.make_layer(current_chs, cfg["mid"])
        self.mid = mid

        ups = []
        decoders = []
        for idx, d in enumerate(cfg["decoder"]):
            align_corners = True if 'bilinear' in up_mode else None
            ups.append(_up_layer(in_chs=current_chs, mode=up_mode, align_corners=align_corners, **kwargs))
            decoder, current_chs = self.make_layer(skip_chs[idx] + current_chs, d)
            decoders.append(decoder)
        self.ups = nn.ModuleList(ups)
        self.decoders = nn.ModuleList(decoders)
        self.predict = nn.Conv2d(current_chs, out_chs, kernel_size=1, stride=1, padding=0, bias=True)
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        encoder_feats = []
        for e in self.encoders:
            x = e(x)
            encoder_feats.insert(0, x)
        x = self.mid(x)
        for f, u, d in zip(encoder_feats, self.ups, self.decoders):
            x = torch.cat((u(x), f), dim=1)
            x = d(x)
        return self.predict(x)

    def make_layer(self, in_chs, cfg):
        layers = []
        current_chs = in_chs
        for v in cfg:
            if v == "max":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2), ]
            elif v == "avg":
                layers += [nn.AvgPool2d(kernel_size=2, stride=2), ]
            elif v == "sconv":
                if self.use_norm:
                    layers += [nn.Conv2d(current_chs, current_chs, kernel_size=3, stride=2, padding=1),
                               _norm_layer(current_chs, **self.kwargs), _act_layer(**self.kwargs)]
                else:
                    layers += [nn.Conv2d(current_chs, current_chs, kernel_size=3, stride=2, padding=1, bias=True),
                               _act_layer(**self.kwargs)]
            elif type(v) is float:
                layers += [nn.Dropout2d(v), ]
            else:
                if self.use_norm:
                    layers += [nn.Conv2d(current_chs, v, kernel_size=3, padding=1),
                               _norm_layer(v, **self.kwargs), _act_layer(**self.kwargs)]
                else:
                    layers += [nn.Conv2d(current_chs, v, kernel_size=3, padding=1, bias=True),
                               _act_layer(**self.kwargs)]
                current_chs = v

        return nn.Sequential(*layers), current_chs


if __name__ == "__main__":
    net = UNet(1, 1, cfg_template, use_norm=True, act_type="relu", norm_type="bn")
    a = torch.rand(2, 1, 64, 64)
    b = net(a)

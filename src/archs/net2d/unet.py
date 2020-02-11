import torch
from torch import nn
import torch.nn.functional as F

from src.archs.net2d.base import _act_layer, _norm_layer, conv1x1, conv3x3, DoubleConv, BasicBlock, Bottleneck, \
    SEBasicBlock, SEBottleneck, ChannelSpatialGate, ChannelSpatialGateV2, block_plus_scse


def _init(net, **kwargs):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            if kwargs.get('act_type') == 'relu':
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            else:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def _up_layer(in_chs=None, mode='conv', align_corners=None, **kwargs):
    assert mode in ['tconv', 'bilinear', 'nearest', 'bilinear-conv', 'nearest-conv'], \
        "Up sampling mode only support 'tconv', 'bilinear', 'nearest', 'bilinear-conv', 'nearest-conv'."
    if mode == 'tconv':
        assert in_chs is not None, "Transposed conv need param in_chs."
        return nn.ConvTranspose2d(in_chs, in_chs, 4, stride=2, padding=1)
    elif len(mode.split('-')) == 2:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode=mode.split('-')[0], align_corners=align_corners),
            conv3x3(in_chs, in_chs),
            _norm_layer(in_chs, **kwargs),
            _act_layer(**kwargs),
        )
    else:
        return nn.Upsample(scale_factor=2, mode=mode, align_corners=align_corners)


def _predict_head(in_chs, out_chs, **kwargs):
    return nn.Sequential(conv3x3(in_chs, in_chs),
                         _norm_layer(in_chs, **kwargs),
                         _act_layer(**kwargs),
                         nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0, bias=True),
                         )


def _repeat(v, length):
    if isinstance(v, (tuple, list)):
        assert len(v) == length, "Input value has different length."
        return tuple(v)
    else:
        return (v,) * length


class UNet(nn.Module):

    def __init__(self, in_chs, out_chs, block, feats, layers, stride=2, groups=1, dilation=1, dropout=None, **kwargs):
        assert isinstance(out_chs, (int, tuple))
        super(UNet, self).__init__()
        up_mode = kwargs.pop('up_mode') if 'up_mode' in kwargs else 'tconv'
        self.kwargs = kwargs
        num_layers = len(feats) - 1
        layers = _repeat(layers, num_layers)
        stride = _repeat(stride, num_layers)
        groups = _repeat(groups, num_layers)
        dilation = _repeat(dilation, num_layers)
        dropout = _repeat(dropout, num_layers * 2)
        if type(block) != tuple:
            block = (block, block)

        self.stem = nn.Sequential(
            conv3x3(in_chs, feats[0], stride=1),
            _norm_layer(feats[0], **self.kwargs),
            _act_layer(**self.kwargs),
            conv3x3(feats[0], feats[0], stride=1),
            _norm_layer(feats[0], **self.kwargs),
            _act_layer(**self.kwargs),
            conv3x3(feats[0], feats[0], stride=1),
            _norm_layer(feats[0], **self.kwargs),
            _act_layer(**self.kwargs),
        )

        encoders = []
        for idx in range(num_layers):
            encoders.append(
                self._make_layer(feats[idx], feats[idx+1], block[0], layers[idx], stride=stride[idx],
                                 groups=groups[idx], dilation=dilation[idx], dropout=dropout[idx], **self.kwargs))
        self.encoders = nn.ModuleList(encoders)

        ups = []
        decoders = []
        for idx in range(num_layers, 0, -1):
            align_corners = True if 'bilinear' in up_mode else None
            ups.append(nn.Sequential(
                _up_layer(in_chs=feats[idx], mode=up_mode, align_corners=align_corners, **kwargs),
                conv1x1(feats[idx], feats[idx - 1]),
                _norm_layer(feats[idx - 1], **self.kwargs),
                _act_layer(**self.kwargs),
            ))
            decoders.append(self._make_layer(feats[idx - 1]*2, feats[idx - 1], block[1], 1, stride=1,
                                             groups=1, dilation=1, dropout=dropout[-idx], **self.kwargs))
        self.ups = nn.ModuleList(ups)
        self.decoders = nn.ModuleList(decoders)

        if isinstance(out_chs, int):
            self.predict = nn.Conv2d(feats[0], out_chs, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            predicts = []
            for o in out_chs:
                predicts.append(_predict_head(feats[0], o, **self.kwargs))
            self.predict = nn.ModuleList(predicts)

    @staticmethod
    def _make_layer(in_chs, out_chs, block, num_blocks, stride=1, groups=1, dilation=1, dropout=None, **kwargs):

        if stride != 1:
            downsample = nn.Sequential(
                nn.AvgPool2d(2, stride=stride),
                conv1x1(in_chs, out_chs),
                _norm_layer(out_chs, **kwargs),
            )
        elif in_chs != out_chs:
            downsample = nn.Sequential(
                conv1x1(in_chs, out_chs),
                _norm_layer(out_chs, **kwargs),
            )
        else:
            downsample = None

        layers = [block(in_chs=in_chs, out_chs=out_chs, stride=stride, downsample=downsample, groups=groups,
                        dilation=dilation if stride == 1 else 1, **kwargs), ]

        for _ in range(1, num_blocks):
            layers.append(block(in_chs=out_chs, out_chs=out_chs, stride=1, groups=groups, dilation=dilation, **kwargs))

        if dropout is not None:
            layers.append(nn.Dropout2d(dropout))

        return nn.Sequential(*layers)

    def forward(self, x):
        # src_size = x.size()[2:]
        y = self.stem(x)
        encoder_feats = [y, ]
        for e in self.encoders:
            y = e(y)
            encoder_feats.insert(0, y)
        for f, u, d in zip(encoder_feats[1:], self.ups, self.decoders):
            y = torch.cat((u(y), f), dim=1)
            y = d(y)
        # y = F.upsample(y, size=src_size, mode='bilinear', align_corners=True)
        if type(self.predict) == nn.ModuleList:
            return [p(y) for p in self.predict]
        else:
            return self.predict(y)


def build_net(in_chs, out_chs, block_type, feats, layers, stride=2, groups=1, dilation=1, dropout=None, **kwargs):
    block_dict = {
        "2conv": DoubleConv,
        "res-a": BasicBlock,
        "res-b": Bottleneck,
        "se-a": SEBasicBlock,
        "se-b": SEBottleneck,
        "res-b-scsev2": block_plus_scse(Bottleneck, ChannelSpatialGateV2),
    }
    if isinstance(block_type, (tuple, list)):
        block = tuple([block_dict[t] for t in block_type])
    else:
        block = block_dict[block_type]
    return UNet(in_chs, out_chs, block, feats=feats, layers=layers, stride=stride, groups=groups, dilation=dilation,
                dropout=dropout, **kwargs)

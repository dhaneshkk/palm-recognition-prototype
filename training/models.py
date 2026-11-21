"""Backbones and margin heads for train.py.

* ResNet-18: global-average-pooled 512-d features (the prototype's encoder).
* MobileFaceNet (Chen et al. 2018): depthwise bottlenecks, ~1 M parameters, and a global depthwise
  convolution (GDConv) head instead of average pooling, so the embedding can weight where on the
  palm a line is. Adapted to the 138×138 ROI (final map 9×9) and a 512-d embedding, the Rust contract.
* Sub-center ArcFace (Deng et al. 2020): K sub-centres per palm, additive angular margin m, scale s.
  K = 1 is plain ArcFace. The margin is ramped up over the first epochs (`set_margin`) because a full
  margin from step one often fails to converge on few-hundred-identity datasets.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _warn(msg):
    from common import log
    log("warning", msg=msg)


class ResNet18(nn.Module):
    out_dim = 512

    def __init__(self, pretrained=True):
        super().__init__()
        r = None
        if pretrained:
            try:
                r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            except Exception as e:  # offline
                _warn(f"ImageNet weights unavailable ({type(e).__name__}); training from scratch")
        r = r or torchvision.models.resnet18(weights=None)
        self.body = nn.Sequential(*list(r.children())[:-1])

    def forward(self, x):
        return self.body(x).flatten(1)


def _cbp(i, o, k=3, s=1, p=1, g=1, act=True):
    layers = [nn.Conv2d(i, o, k, s, p, groups=g, bias=False), nn.BatchNorm2d(o)]
    if act:
        layers.append(nn.PReLU(o))
    return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    def __init__(self, i, o, s, t):
        super().__init__()
        h = i * t
        self.res = s == 1 and i == o
        self.f = nn.Sequential(_cbp(i, h, 1, 1, 0), _cbp(h, h, 3, s, 1, g=h), _cbp(h, o, 1, 1, 0, act=False))

    def forward(self, x):
        return x + self.f(x) if self.res else self.f(x)


class MobileFaceNet(nn.Module):
    out_dim = 512

    def __init__(self, side=138, emb=512):
        super().__init__()
        cfg = [(2, 64, 5, 2), (4, 128, 1, 2), (2, 128, 6, 1), (4, 128, 1, 2), (2, 128, 2, 1)]  # t, c, n, s
        layers = [_cbp(3, 64, 3, 2, 1), _cbp(64, 64, 3, 1, 1, g=64)]
        c_in = 64
        for t, c, n, s in cfg:
            for j in range(n):
                layers.append(Bottleneck(c_in, c, s if j == 0 else 1, t))
                c_in = c
        layers.append(_cbp(c_in, 512, 1, 1, 0))
        self.body = nn.Sequential(*layers)
        with torch.no_grad():
            fh = self.body(torch.zeros(1, 3, side, side)).shape[-1]
        self.gdconv = nn.Sequential(nn.Conv2d(512, 512, fh, 1, 0, groups=512, bias=False), nn.BatchNorm2d(512))
        self.fc = nn.Sequential(nn.Linear(512, emb, bias=False), nn.BatchNorm1d(emb))

    def forward(self, x):
        return self.fc(self.gdconv(self.body(x)).flatten(1))


def backbone(name, pretrained=True):
    if name == "resnet18":
        return ResNet18(pretrained)
    if name == "mobilefacenet":
        if pretrained:
            _warn("no ImageNet weights for MobileFaceNet; training from scratch")
        return MobileFaceNet()
    raise ValueError(name)


class SubCenterArcFace(nn.Module):
    """Logits for cross-entropy: s · cos(θ_y + m) for the true palm, s · cos θ_j otherwise, where
    cos θ_j is the best of palm j's K sub-centres."""

    def __init__(self, dim, classes, k=3, s=30.0, m=0.5):
        super().__init__()
        self.k, self.s, self.m_max, self.m = k, s, m, 0.0
        self.w = nn.Parameter(torch.empty(classes * k, dim))
        nn.init.xavier_uniform_(self.w)

    def set_margin(self, m):
        self.m = float(min(max(m, 0.0), self.m_max))

    def forward(self, emb, labels):
        cos = F.linear(F.normalize(emb), F.normalize(self.w)).view(emb.shape[0], -1, self.k).amax(dim=2)
        cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
        if self.m > 0:
            theta = torch.acos(cos.gather(1, labels[:, None]))
            target = torch.cos(theta + self.m)
            # past π − m the margin would make the logit grow again: use the usual linear fallback
            target = torch.where(theta + self.m < math.pi, target, cos.gather(1, labels[:, None]) - self.m * math.sin(self.m))
            cos = cos.scatter(1, labels[:, None], target)
        return F.cross_entropy(self.s * cos, labels)

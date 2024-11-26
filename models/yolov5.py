

import torch
import torch.nn as nn
from models.common import (
    Conv,
    DWConv,
    Proto,
)
from utils.general import check_version
import math

class DecoupledDetect(nn.Module):
    """YOLOv5 Detect head for processing input tensors and generating detection outputs in object detection models."""

    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        """Initializes YOLOv5 detection layer with specified classes, anchors, channels, and inplace operations."""
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        
        # 方法1
        self.mloc = nn.ModuleList(
            nn.Conv2d(x, 4 * self.na, 1) for x in ch
        )
        self.mcls = nn.ModuleList(
            nn.Sequential(DWConv(x, x, 3), nn.Conv2d(x, self.nc*self.na, 1))
            for x in ch
        )
        self.mobj = nn.ModuleList(
            nn.Sequential(DWConv(x, x, 3), nn.Conv2d(x, 1*self.na, 1))
            for x in ch
        )

        # 方法2
        # self.mloc = nn.ModuleList(nn.Conv2d(x, 4 * self.na, 1) for x in ch)
        # self.mcls = nn.ModuleList(nn.Conv2d(x, self.nc*self.na, 1)for x in ch)
        # self.mobj = nn.ModuleList(nn.Conv2d(x, 1*self.na, 1)for x in ch)

        self.m = [self.mloc, self.mobj, self.mcls]
        
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.no_decode = False

    def forward(self, x):
        """Processes input through YOLOv5 layers, altering shape for detection: `x(bs, 3, ny, nx, 85)`."""
        z = []  # inference output
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = torch.concat([m[i](x[i]).view(bs, self.na, -1, ny, nx).permute(0, 1, 3, 4, 2).contiguous() 
                                 for m in self.m], -1)

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, DecoupledSegment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training or self.no_decode else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid
    
    def bias_init(self, cf=None):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        for i in range(len(self.stride)):
            self.mloc[i].bias.data[:] = 1
            # 方法1
            self.mobj[i][-1].bias.data[:] = math.log(8 / (640 / self.stride[i]) ** 2)  # obj (8 objects per 640 image)
            self.mcls[i][-1].bias.data[:] = math.log(0.6 / (self.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())

            # 方法2
            # self.mobj[i].bias.data += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            # self.mcls[i].bias.data += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
 


class DecoupledSegment(DecoupledDetect):
    """YOLOv5 Segment head for segmentation models, extending Detect with mask and prototype layers."""

    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        """Initializes YOLOv5 Segment head with options for mask count, protos, and channel adjustments."""
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        c2, c3 = max((16, ch[0] // 4, 16 * 4))
        self.mseg = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, self.nm  * self.na, 1)) for x in ch
        )
        self.m.append(self.mseg)
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = DecoupledDetect.forward

    def forward(self, x):
        """Processes input through the network, returning detections and prototypes; adjusts output based on
        training/export mode.
        """
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])

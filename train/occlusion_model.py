
import torch, torch.nn as nn, torch.nn.functional as F

OCCLUSION_CLASSES = ["mask","hand","phone","glasses","scarf","hat","image","unknown"]
PARTS = ["eyes","nose","mouth","cheeks","forehead"]

class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None):
        super().__init__()
        p = k//2 if p is None else p
        self.cv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.cv(x)))

class OcclusionNet(nn.Module):
    def __init__(self, num_classes=len(OCCLUSION_CLASSES), parts=len(PARTS), img=256):
        super().__init__()
        c = 32
        self.backbone = nn.Sequential(
            ConvBNAct(3, c, 3, 2),
            ConvBNAct(c, c, 3, 1),
            ConvBNAct(c, 2*c, 3, 2),
            ConvBNAct(2*c, 2*c, 3, 1),
            ConvBNAct(2*c, 4*c, 3, 2),
            ConvBNAct(4*c, 4*c, 3, 1),
            ConvBNAct(4*c, 8*c, 3, 2),
            ConvBNAct(8*c, 8*c, 3, 1),
        )
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8*c, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(8*c, 4*c, 2, 2),
            ConvBNAct(4*c, 2*c, 3, 1),
            nn.ConvTranspose2d(2*c, c, 2, 2),
            ConvBNAct(c, c, 3, 1),
            nn.ConvTranspose2d(c, c, 2, 2),
            ConvBNAct(c, c, 3, 1),
            nn.ConvTranspose2d(c, parts, 2, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        f = self.backbone(x)
        cls = self.cls_head(f)
        seg = self.seg_head(f)
        return cls, seg

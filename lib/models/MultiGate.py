import torch
from torch import tensor
import torch.nn as nn
import sys, os
import math
import sys

sys.path.append(os.getcwd())
from lib.utils import initialize_weights
from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv, Conv_decoder, C3TR, ELANBlock
from lib.models.common_Multi import gateAttention
# from lib.models.common_hift import HierarchicalFFM, CrossLayerAttention
from lib.models.common_hift_pre import HierarchicalFFM, HierarchicalFFM2, CrossLayerAttention, CrossLayerAttention_noheight
from lib.models.common_hanet import HCAModule
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized

MultiGate = [
[20, 30, 40],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, C3TR, [512, 512, 1, False]],     #9
[ [4, 6], HierarchicalFFM, [128, 256]],    #10

[ [10, 9], CrossLayerAttention, [128, 512, True]],    #11
[ -1, Conv, [128, 128, 3, 2]],      #12
[ 6, Conv,[256, 128, 1, 1]],     #13
[ [-1, 12], Concat, [1]],       #14
[ -1, BottleneckCSP, [256, 256, 1, False]],     #15
[ -1, Conv, [256, 256, 3, 2]],      #16
[ 9, Conv,[512, 256, 1, 1]],   #17
[ [-1, 16], Concat, [1]],   #18
[ -1, BottleneckCSP, [512, 512, 1, False]],     #19
[ [11, 15, 19], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detection head 20

[ [10, 9], CrossLayerAttention, [128, 512, True]], #21
[ -1, Conv, [128, 128, 3, 1]],   #22
[ -1, Upsample, [None, 2, 'nearest']],  #23
[ -1, BottleneckCSP, [128, 64, 1, False]],  #24
[ -1, Conv, [64, 32, 3, 1]],    #25
[ -1, Upsample, [None, 2, 'nearest']],  #26
[ -1, Conv, [32, 16, 3, 1]],    #27
[ -1, BottleneckCSP, [16, 8, 1, False]],    #28
[ -1, Upsample, [None, 2, 'nearest']],  #29
[ -1, Conv, [8, 2, 3, 1]], #30 Driving area segmentation head

[ [10, 9], CrossLayerAttention, [128, 512, True]], #31
[ -1, Conv, [128, 128, 3, 1]],   #32
[ -1, Upsample, [None, 2, 'nearest']],  #33
[ -1, BottleneckCSP, [128, 64, 1, False]],  #34
[ -1, Conv, [64, 32, 3, 1]],    #35
[ -1, Upsample, [None, 2, 'nearest']],  #36
[ -1, Conv, [32, 16, 3, 1]],    #37
[ -1, BottleneckCSP, [16, 8, 1, False]],    #38
[ -1, Upsample, [None, 2, 'nearest']],  #39
[ -1, Conv, [8, 2, 3, 1]] #40 Lane line segmentation head
]


YOLOP_hffm = [
[20, 29, 38],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ [6, -1], HierarchicalFFM, [256, 512]],    #9
[ [4, -1], HierarchicalFFM, [128, 256]],    #10

[ -1, BottleneckCSP, [128, 128, 1, False]],     #11
[ -1, Conv, [128, 128, 3, 2]],      #12
[ 9, Conv, [256, 128, 3, 1]],      #13
[ [-1, 12], Concat, [1]],       #14
[ -1, BottleneckCSP, [256, 256, 1, False]],     #15
[ -1, Conv, [256, 256, 3, 2]],      #16
[ 8,  Conv, [512, 256, 1, 1]],   #17
[ [-1, 16], Concat, [1]],   #18
[ -1, BottleneckCSP, [512, 512, 1, False]],     #19
[ [11, 15, 19], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detection head 20

[ 10, Conv, [128, 128, 3, 1]],   #21
[ -1, Upsample, [None, 2, 'nearest']],  #22
[ -1, BottleneckCSP, [128, 64, 1, False]],  #23
[ -1, Conv, [64, 32, 3, 1]],    #24
[ -1, Upsample, [None, 2, 'nearest']],  #25
[ -1, Conv, [32, 16, 3, 1]],    #26
[ -1, BottleneckCSP, [16, 8, 1, False]],    #27
[ -1, Upsample, [None, 2, 'nearest']],  #28
[ -1, Conv, [8, 2, 3, 1]], #29 Driving area segmentation head

[ 10, Conv, [128, 128, 3, 1]],   #30
[ -1, Upsample, [None, 2, 'nearest']],  #31
[ -1, BottleneckCSP, [128, 64, 1, False]],  #32
[ -1, Conv, [64, 32, 3, 1]],    #33
[ -1, Upsample, [None, 2, 'nearest']],  #34
[ -1, Conv, [32, 16, 3, 1]],    #35
[ -1, BottleneckCSP, [16, 8, 1, False]],    #36
[ -1, Upsample, [None, 2, 'nearest']],  #37
[ -1, Conv, [8, 2, 3, 1]] #38 Lane line segmentation head
]


YOLOP_hca = [
[25, 35, 45],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16         #Encoder
[ -1, BottleneckCSP, [256, 128, 1, False]],     #17

[ [17, 9], CrossLayerAttention, [128, 512, True]], #18
[ -1, Conv, [128, 128, 3, 2]],      #19
[ [-1, 14], Concat, [1]],       #20
[ -1, BottleneckCSP, [256, 256, 1, False]],     #21
[ -1, Conv, [256, 256, 3, 2]],      #22
[ [-1, 10], Concat, [1]],   #23
[ -1, BottleneckCSP, [512, 512, 1, False]],     #24
[ [17, 21, 24], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detection head 24

[ [17, 9], CrossLayerAttention, [128, 512, True]], #26
[ -1, Conv, [128, 128, 3, 1]],   #27
[ -1, Upsample, [None, 2, 'nearest']],  #28
[ -1, BottleneckCSP, [128, 64, 1, False]],  #29
[ -1, Conv, [64, 32, 3, 1]],    #30
[ -1, Upsample, [None, 2, 'nearest']],  #31
[ -1, Conv, [32, 16, 3, 1]],    #32
[ -1, BottleneckCSP, [16, 8, 1, False]],    #33
[ -1, Upsample, [None, 2, 'nearest']],  #34
[ -1, Conv, [8, 2, 3, 1]], #35 Driving area segmentation head

[ [17, 9], CrossLayerAttention, [128, 512, True]], #36
[ -1, Conv, [128, 128, 3, 1]],   #37
[ -1, Upsample, [None, 2, 'nearest']],  #38
[ -1, BottleneckCSP, [128, 64, 1, False]],  #39
[ -1, Conv, [64, 32, 3, 1]],    #40
[ -1, Upsample, [None, 2, 'nearest']],  #41
[ -1, Conv, [32, 16, 3, 1]],    #42
[ -1, BottleneckCSP, [16, 8, 1, False]],    #43
[ -1, Upsample, [None, 2, 'nearest']],  #44
[ -1, Conv, [8, 2, 3, 1]] #45 Lane line segmentation head
]


YOLOP_hca_noheight = [
[25, 35, 45],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16         #Encoder
[ -1, BottleneckCSP, [256, 128, 1, False]],     #17

[ [17, 9], CrossLayerAttention_noheight, [128, 512, True]], #18
[ -1, Conv, [128, 128, 3, 2]],      #19
[ [-1, 14], Concat, [1]],       #20
[ -1, BottleneckCSP, [256, 256, 1, False]],     #21
[ -1, Conv, [256, 256, 3, 2]],      #22
[ [-1, 10], Concat, [1]],   #23
[ -1, BottleneckCSP, [512, 512, 1, False]],     #24
[ [17, 21, 24], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detection head 24

[ [17, 9], CrossLayerAttention_noheight, [128, 512, True]], #26
[ -1, Conv, [128, 128, 3, 1]],   #27
[ -1, Upsample, [None, 2, 'nearest']],  #28
[ -1, BottleneckCSP, [128, 64, 1, False]],  #29
[ -1, Conv, [64, 32, 3, 1]],    #30
[ -1, Upsample, [None, 2, 'nearest']],  #31
[ -1, Conv, [32, 16, 3, 1]],    #32
[ -1, BottleneckCSP, [16, 8, 1, False]],    #33
[ -1, Upsample, [None, 2, 'nearest']],  #34
[ -1, Conv, [8, 2, 3, 1]], #35 Driving area segmentation head

[ [17, 9], CrossLayerAttention_noheight, [128, 512, True]], #36
[ -1, Conv, [128, 128, 3, 1]],   #37
[ -1, Upsample, [None, 2, 'nearest']],  #38
[ -1, BottleneckCSP, [128, 64, 1, False]],  #39
[ -1, Conv, [64, 32, 3, 1]],    #40
[ -1, Upsample, [None, 2, 'nearest']],  #41
[ -1, Conv, [32, 16, 3, 1]],    #42
[ -1, BottleneckCSP, [16, 8, 1, False]],    #43
[ -1, Upsample, [None, 2, 'nearest']],  #44
[ -1, Conv, [8, 2, 3, 1]] #45 Lane line segmentation head
]


class MCnet(nn.Module):
    def __init__(self, block_cfg, **kwargs):
        super(MCnet, self).__init__()
        layers, save = [], []
        self.nc = 1
        self.detector_index = -1
        self.det_out_idx = block_cfg[0][0]
        self.seg_out_idx = block_cfg[0][1:]

        # Build model
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is Detect:
                self.detector_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
        # assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                detects, _, _ = model_out
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()

        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        det_out = None
        Da_fmap = []
        LL_fmap = []
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in
                                                                             block.from_]  # calculate concat detect
            x = block(x)
            if i in self.seg_out_idx:  # save driving area segment result
                m = nn.Sigmoid()
                out.append(m(x))
            if i == self.detector_index:
                det_out = x
            cache.append(x if block.index in self.save else None)
        out.insert(0, det_out)
        return out

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def get_net(cfg, **kwargs):
    m_block_cfg = YOLOP_hca
    model = MCnet(m_block_cfg, **kwargs)
    return model


if __name__ == "__main__":

    model = get_net(False)
    model_state= model.state_dict()
    input = torch.randn((1, 3, 256, 256))
    gt = torch.rand((1, 2, 256, 256))

    detects, dring_area_seg, lane_line_seg = model(input)
    for det in detects:
        print(det.shape)
    print(dring_area_seg.shape)
    print(lane_line_seg.shape)
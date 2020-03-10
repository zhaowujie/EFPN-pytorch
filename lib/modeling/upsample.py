import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import nn as mynn
import numpy as np
from torch.autograd import Variable

class SR(nn.Module):
    def __init__(self):
        super(SR, self).__init__()
        self.blocks =  nn.ModuleList()
        self.blocks.append(Extension(2))

    def detectron_weight_mapping(self):
        mapping_to_detectron = {}
        return mapping_to_detectron, []



class Extension(nn.Module):
    def __init__(self, upscale_factor):
        super(Extension, self).__init__()
        self.upscale_factor = upscale_factor


        self.FTT = FeatureTextureTransfer(ngf=512)
        self.conv_lateral = nn.Conv2d(256, 256, 1, 1, 0, bias=False)
        self.post = nn.Conv2d(256, 256, 3, 1, 1, bias=False)

        self._initialize_weights()

    def forward(self, fpn_inner3, fpn_inner2, C1_2):

        # Feature Texture Transfer
        fpn_inner3_upsample = F.upsample(fpn_inner3, scale_factor=self.upscale_factor, mode='nearest')
        fpn_inner3_sr = self.FTT(fpn_inner3, fpn_inner2) + fpn_inner3_upsample

        # FPN-like top-down & lateral connection
        fpn_inner3_sr_upsample = F.upsample(fpn_inner3_sr, scale_factor=2, mode='nearest')
        lat = self.conv_lateral(C1_2)
        P2_sr = self.post(lat + fpn_inner3_sr_upsample)

        return P2_sr, fpn_inner3_sr


    def _initialize_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.XavierFill(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        self.apply(_init)
    def detectron_weight_mapping(self):
        mapping_to_detectron = {}
        return mapping_to_detectron, []



class FeatureTextureTransfer(nn.Module):
    """
    Conditional Texture Transfer for SRNTT,
        see https://github.com/ZZUTK/SRNTT/blob/master/SRNTT/model.py#L116.
    This module is devided 3 parts for each scales.

    Parameters
    ---
    ngf : int
        a number of generator's filters.
    n_blocks : int, optional
        a number of residual blocks, see also `ResBlock` class.
    """

    def __init__(self, ngf):
        super(FeatureTextureTransfer, self).__init__()

        self.content_extractor = ContentExtractor(ngf=ngf)  #
        self.sub_pixel = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
        )
        self.texture_extractor = TextureExtractor(ngf=ngf)

        self.tail = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf, 256, kernel_size=3, stride=1, padding=1)
        )


    def forward(self, fpn_inner3, fpn_inner2):
        # main stream
        content = self.content_extractor(fpn_inner3)
        content = self.sub_pixel(content)
        # reference stream
        wrap = torch.cat([content, fpn_inner2], 1)
        fpn_inner3_sr = self.texture_extractor(wrap) + content
        fpn_inner3_sr = self.tail(fpn_inner3_sr)

        return fpn_inner3_sr


class ContentExtractor(nn.Module):
    """
    Content Extractor for SRNTT, which outputs maps before-and-after upscale.
    more detail: https://github.com/ZZUTK/SRNTT/blob/master/SRNTT/model.py#L73.
    Currently this module only supports `scale_factor=4`.

    Parameters
    ---
    ngf : int, optional
        a number of generator's features.
    n_blocks : int, optional
        a number of residual blocks, see also `ResBlock` class.
    """

    def __init__(self, ngf, n_blocks=1):
        super(ContentExtractor, self).__init__()

        self.content_head = nn.Sequential(
            nn.Conv2d(256, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
        )
        self.content_body = nn.Sequential(
            *[ResBlock(ngf) for _ in range(n_blocks)],
            # nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(ngf)
        )

    def forward(self, x):
        h = self.content_head(x)
        h = self.content_body(h) + h
        return h

class TextureExtractor(nn.Module):
    """
    Content Extractor for SRNTT, which outputs maps before-and-after upscale.
    more detail: https://github.com/ZZUTK/SRNTT/blob/master/SRNTT/model.py#L73.
    Currently this module only supports `scale_factor=4`.

    Parameters
    ---
    ngf : int, optional
        a number of generator's features.
    n_blocks : int, optional
        a number of residual blocks, see also `ResBlock` class.
    """

    def __init__(self, ngf, n_blocks=2):
        super(TextureExtractor, self).__init__()

        self.texture_head = nn.Sequential(
            nn.Conv2d(ngf + 256, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
        )
        self.texture_body = nn.Sequential(
            *[ResBlock(ngf) for _ in range(n_blocks)],
        )

    def forward(self, x):
        h = self.texture_head(x)
        h = self.texture_body(h)
        return h



class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel_num, channel_num, 3, padding=1)
        )

        self.relu = nn.ReLU()
        self._initialize_weights()

    def forward(self, x):
        residual = x
        x = self.conv_block(x)
        x = x + residual
        out = self.relu(x)
        return out

    def _initialize_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)
        self.apply(_init)



def compute_SR_patchloss(pred_map, target_map, rois, sc):

    rois = rois_reshape(rois, 1 / sc)
    device_id = pred_map.get_device()
    weight_map = generate_roi_weight(rois, target_map, sc)
    weight_map = Variable(weight_map).cuda(device_id)

    loss = F.l1_loss(pred_map * weight_map, target_map * weight_map,
                           size_average=False) / weight_map.sum()
    return loss


def compute_SR_loss(pred_map, target_map):


    loss = F.l1_loss(pred_map, target_map)
    return loss

def feature_roi_crop(rois, conv, spatial_scale):
    """

    :param rois: x1y1x2y2
    :param conv: batch_size * channels * height * width
    :return:
    """
    assert len(conv.shape)==4
    batch_size = conv.shape[0]
    blobs_out = []
    for batch_idx in range(batch_size):
        rois_idx = np.where(rois[:, 0] == batch_idx)[0]
        rois_num = len(rois_idx)
        rois_batch = rois[rois_idx, 1:5] * spatial_scale
        rois_batch = rois_batch.astype(np.int32)
        for roi in rois_batch:
            roi[0] = max(0, roi[0])
            roi[1] = max(0, roi[1])
            roi[2] = min(conv.shape[3], roi[2])
            roi[3] = min(conv.shape[2], roi[3])
            blobs_out.append(conv[batch_idx:batch_idx+1, :, roi[1]:roi[3], roi[0]:roi[2]])
    return blobs_out

def rois_reshape(rois, mod):
    rois_int = np.zeros(rois.shape).astype(np.int32)

    rois_int[:, 0] = rois[:, 0]
    rois_int[:, 1] = np.floor((rois[:, 1]) / mod) * mod
    rois_int[:, 2] = np.floor((rois[:, 2]) / mod) * mod
    rois_int[:, 3] = np.ceil((rois[:, 3]) / mod) * mod
    rois_int[:, 4] = np.ceil((rois[:, 4]) / mod) * mod
    area = (rois_int[:, 4] - rois_int[:, 2]) * (rois_int[:, 3] - rois_int[:, 1])
    keep_idx = np.where(area != 0)[0]
    return rois_int[keep_idx, :]

def generate_roi_weight(rois, conv, spatial_scale):
    """

        :param rois: x1y1x2y2
        :param conv: batch_size * channels * height * width
        :return:
    """
    assert len(conv.shape) == 4
    batch_size = conv.shape[0]
    device_id = conv.get_device()
    weights1 = torch.zeros((conv.shape)).cuda(device_id)
    # weights2 = torch.zeros((conv.shape[0], conv.shape[2], conv.shape[3])).cuda(device_id)

    for batch_idx in range(batch_size):
        rois_idx = np.where(rois[:, 0] == batch_idx)[0]
        rois_batch = rois[rois_idx, 1:5] * spatial_scale
        rois_batch = rois_batch.astype(np.int32)
        for roi in rois_batch:
            roi[0] = max(0, roi[0])
            roi[1] = max(0, roi[1])
            roi[2] = min(conv.shape[3], roi[2])
            roi[3] = min(conv.shape[2], roi[3])
            weights1[batch_idx, :, roi[1]:roi[3], roi[0]:roi[2]] = 1
            # weights2[batch_idx,  roi[1]:roi[3], roi[0]:roi[2]] = 1
    # print('weights1:', weights1.mean())
    # print('weights2:', weights2.mean())

    return weights1


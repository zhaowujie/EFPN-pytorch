import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
import nn as mynn
import utils.net as net_utils
import numpy as np


class fast_rcnn_outputs(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.cls_score = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES)
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:  # bg and fg
            self.bbox_pred = nn.Linear(dim_in, 4 * 2)
        else:
            self.bbox_pred = nn.Linear(dim_in, 4 * cfg.MODEL.NUM_CLASSES)

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, 0)
        init.normal_(self.bbox_pred.weight, std=0.001)
        init.constant_(self.bbox_pred.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'cls_score.weight': 'cls_score_w',
            'cls_score.bias': 'cls_score_b',
            'bbox_pred.weight': 'bbox_pred_w',
            'bbox_pred.bias': 'bbox_pred_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        cls_score = self.cls_score(x)
        if not self.training:
            cls_score = F.softmax(cls_score, dim=1)
        bbox_pred = self.bbox_pred(x)

        return cls_score, bbox_pred

class bbox_outputs(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.bbox_pred = nn.Linear(dim_in, 4)
        self.cls_score = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES)


        self._init_weights()

    def _init_weights(self):
        init.normal_(self.bbox_pred.weight, std=0.001)
        init.constant_(self.bbox_pred.bias, 0)
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'bbox_pred.weight': 'bbox_pred_w',
            'bbox_pred.bias': 'bbox_pred_b',
            'cls_score.weight': 'cls_score_w',
            'cls_score.bias': 'cls_score_b'

        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        bbox_pred = self.bbox_pred(x)
        cls_score = self.cls_score(x)
        if not self.training:
            cls_score = F.softmax(cls_score, dim=1)

        return bbox_pred, cls_score

class cls_outputs(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.cls_score = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES)
        self._init_weights()

    def _init_weights(self):
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'cls_score.weight': 'cls_score_w',
            'cls_score.bias': 'cls_score_b',
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        cls_score = self.cls_score(x)
        if not self.training:
            cls_score = F.softmax(cls_score, dim=1)

        return cls_score


def fast_rcnn_losses(cls_score, bbox_pred, label_int32, bbox_targets,
                     bbox_inside_weights, bbox_outside_weights):
    device_id = cls_score.get_device()
    rois_label = Variable(torch.from_numpy(label_int32.astype('int64'))).cuda(device_id)
    # print(cls_score.shape, rois_label.shape) # torch.Size([512, 81]) torch.Size([512])
    # print(bbox_pred.shape, bbox_targets.shape) # torch.Size([512, 324]) (512, 324)


    loss_cls = F.cross_entropy(cls_score, rois_label)

    bbox_targets = Variable(torch.from_numpy(bbox_targets)).cuda(device_id)
    bbox_inside_weights = Variable(torch.from_numpy(bbox_inside_weights)).cuda(device_id)
    bbox_outside_weights = Variable(torch.from_numpy(bbox_outside_weights)).cuda(device_id)
    loss_bbox = net_utils.smooth_l1_loss(
        bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

    # class accuracy
    cls_preds = cls_score.max(dim=1)[1].type_as(rois_label)
    accuracy_cls = cls_preds.eq(rois_label).float().mean(dim=0)


    return loss_cls, loss_bbox, accuracy_cls

def bbox_losses(bbox_pred, bbox_targets,
                     bbox_inside_weights, bbox_outside_weights):
    device_id = bbox_pred.get_device()


    bbox_targets = Variable(torch.from_numpy(bbox_targets)).cuda(device_id)
    bbox_inside_weights = Variable(torch.from_numpy(bbox_inside_weights)).cuda(device_id)
    bbox_outside_weights = Variable(torch.from_numpy(bbox_outside_weights)).cuda(device_id)
    loss_bbox = net_utils.smooth_l1_loss(
        bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)



    return loss_bbox


def cls_losses(cls_score, label_int32):
    device_id = cls_score.get_device()
    rois_label = Variable(torch.from_numpy(label_int32.astype('int64'))).cuda(device_id)

    loss_cls = F.cross_entropy(cls_score, rois_label)


    # class accuracy
    cls_preds = cls_score.max(dim=1)[1].type_as(rois_label)
    accuracy_cls = cls_preds.eq(rois_label).float().mean(dim=0)


    return loss_cls, accuracy_cls

# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #

class roi_2mlp_head(nn.Module):
    """Add a ReLU MLP with two hidden layers."""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM

        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc1 = nn.Linear(dim_in * roi_size**2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        mynn.init.XavierFill(self.fc1.weight)
        init.constant_(self.fc1.bias, 0)
        mynn.init.XavierFill(self.fc2.weight)
        init.constant_(self.fc2.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'fc1.weight': 'fc6_w',
            'fc1.bias': 'fc6_b',
            'fc2.weight': 'fc7_w',
            'fc2.bias': 'fc7_b'
        }
        return detectron_weight_mapping, []

    def forward(self, x, rpn_ret, sc=None):
        if sc is not None:
            x = self.roi_xform(
                x, rpn_ret,
                blob_rois='rois',
                method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
                resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
                #spatial_scale=self.spatial_scale,
                spatial_scale=sc,
                sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
            )
        else:
            x = self.roi_xform(
                x, rpn_ret,
                blob_rois='rois',
                method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
                resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
                spatial_scale=self.spatial_scale,
                sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
            )
        batch_size = x.size(0)
        x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        return x


class roi_Xconv1fc_head(nn.Module):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm"""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
        module_list = []
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            module_list.extend([
                nn.Conv2d(dim_in, hidden_dim, 3, 1, 1),
                nn.ReLU(inplace=True)
            ])
            dim_in = hidden_dim
        self.convs = nn.Sequential(*module_list)

        self.dim_out = fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc = nn.Linear(dim_in * roi_size * roi_size, fc_dim)

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.MSRAFill(m.weight)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)
        self.apply(_init)

    def detectron_weight_mapping(self):
        mapping = {}
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            mapping.update({
                'convs.%d.weight' % (i*2): 'head_conv%d_w' % (i+1),
                'convs.%d.bias' % (i*2): 'head_conv%d_b' % (i+1)
            })
        mapping.update({
            'fc.weight': 'fc6_w',
            'fc.bias': 'fc6_b'
        })
        return mapping, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = self.convs(x)
        x = F.relu(self.fc(x.view(batch_size, -1)), inplace=True)
        return x


class roi_Xconv1fc_gn_head(nn.Module):
    """Add a X conv + 1fc head, with GroupNorm"""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
        module_list = []
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            module_list.extend([
                nn.Conv2d(dim_in, hidden_dim, 3, 1, 1, bias=False),
                nn.GroupNorm(net_utils.get_group_gn(hidden_dim), hidden_dim,
                             eps=cfg.GROUP_NORM.EPSILON),
                nn.ReLU(inplace=True)
            ])
            dim_in = hidden_dim
        self.convs = nn.Sequential(*module_list)

        self.dim_out = fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc = nn.Linear(dim_in * roi_size * roi_size, fc_dim)

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.MSRAFill(m.weight)
            elif isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)
        self.apply(_init)

    def detectron_weight_mapping(self):
        mapping = {}
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            mapping.update({
                'convs.%d.weight' % (i*3): 'head_conv%d_w' % (i+1),
                'convs.%d.weight' % (i*3+1): 'head_conv%d_gn_s' % (i+1),
                'convs.%d.bias' % (i*3+1): 'head_conv%d_gn_b' % (i+1)
            })
        mapping.update({
            'fc.weight': 'fc6_w',
            'fc.bias': 'fc6_b'
        })
        return mapping, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = self.convs(x)
        x = F.relu(self.fc(x.view(batch_size, -1)), inplace=True)
        return x


class OHEM_Sampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_expected = cfg.TRAIN.OHEM_SAMPLE_NUM * cfg.TRAIN.IMS_PER_BATCH
        self.num_expected_pos = int(self.num_expected * 0.25)
        self.num_expected_neg = self.num_expected - self.num_expected_pos




    def forward(self, cls_score, bbox_pred, label_int32, bbox_targets,
                     bbox_inside_weights, bbox_outside_weights):


        device_id = cls_score.get_device()
        rois_label = Variable(torch.from_numpy(label_int32.astype('int64')), requires_grad=False).cuda(device_id)
        bbox_targets = Variable(torch.from_numpy(bbox_targets), requires_grad=False).cuda(device_id)
        bbox_inside_weights = Variable(torch.from_numpy(bbox_inside_weights), requires_grad=False).cuda(device_id)
        bbox_outside_weights = Variable(torch.from_numpy(bbox_outside_weights), requires_grad=False).cuda(device_id)

        neg_inds = np.where(label_int32 == 0)
        pos_inds = np.where(label_int32 > 0)

        if len(neg_inds[0]):
            neg_inds = torch.from_numpy(neg_inds[0]).cuda(device_id)
            sample_neg_inds = self.sample_neg(neg_inds, cls_score, rois_label)
        else:
            sample_neg_inds = torch.LongTensor().cuda(device_id)

        if len(pos_inds[0]):
            pos_inds = torch.from_numpy(pos_inds[0]).cuda(device_id)
            sample_pos_inds = self.sample_pos(pos_inds, cls_score, rois_label, bbox_pred, bbox_targets,
                                              bbox_inside_weights,
                                              bbox_outside_weights)
        else:
            sample_pos_inds = torch.LongTensor().cuda(device_id)

        return torch.cat((sample_pos_inds, sample_neg_inds))



    def sample_pos(self, pos_inds, cls_score, rois_label, bbox_pred, bbox_targets, bbox_inside_weights,
                                   bbox_outside_weights):
        if len(pos_inds) < self.num_expected_pos:
            return pos_inds
        else:
            cls_loss = F.cross_entropy(cls_score[pos_inds], rois_label[pos_inds], reduce=False)
            bbox_loss = net_utils.smooth_l1_loss(
                bbox_pred[pos_inds], bbox_targets[pos_inds], bbox_inside_weights[pos_inds],
                bbox_outside_weights[pos_inds], reduce=False)
            loss = cls_loss + bbox_loss
            _, sample_inds = loss.topk(self.num_expected_pos)
            sample_inds = sample_inds.data
            return pos_inds[sample_inds]

    def sample_neg(self, neg_inds, cls_score, rois_label):

        if len(neg_inds) < self.num_expected_neg:
            return neg_inds
        else:
            loss_cls = F.cross_entropy(cls_score[neg_inds], rois_label[neg_inds], reduce=False) # size:[913]
            _, sample_inds = loss_cls.topk(self.num_expected_neg)
            sample_inds = sample_inds.data
            return neg_inds[sample_inds]






from functools import wraps
import importlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg
from model.roi_pooling.functions.roi_pool import RoIPoolFunction
from model.roi_crop.functions.roi_crop import RoICropFunction
from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
import modeling.rpn_heads as rpn_heads
import modeling.fast_rcnn_heads as fast_rcnn_heads
import utils.blob as blob_utils
import utils.net as net_utils
import utils.resnet_weights_helper as resnet_utils
import nn as mynn
import numpy as np
from modeling.upsample import SR, compute_SR_loss, compute_SR_patchloss, Extension


logger = logging.getLogger(__name__)

def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise


def compare_state_dict(sa, sb):
    if sa.keys() != sb.keys():
        return False
    for k, va in sa.items():
        if not torch.equal(va, sb[k]):
            return False
    return True


def check_inference(net_func):
    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        if not self.training:
            if cfg.PYTORCH_VERSION_LESS_THAN_040:
                return net_func(self, *args, **kwargs)
            else:
                with torch.no_grad():
                    return net_func(self, *args, **kwargs)
        else:
            raise ValueError('You should call this function only on inference.'
                             'Set the network in inference mode by net.eval().')

    return wrapper


class Generalized_RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        # Backbone for feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()

        # Region Proposal Network
        if cfg.RPN.RPN_ON:
            self.RPN = rpn_heads.generic_rpn_outputs(
                self.Conv_Body.dim_out, self.Conv_Body.spatial_scale)

        if cfg.FPN.FPN_ON:
            # Only supports case when RPN and ROI min levels are the same
            assert cfg.FPN.RPN_MIN_LEVEL == cfg.FPN.ROI_MIN_LEVEL
            # RPN max level can be >= to ROI max level
            assert cfg.FPN.RPN_MAX_LEVEL >= cfg.FPN.ROI_MAX_LEVEL
            # FPN RPN max level might be > FPN ROI max level in which case we
            # need to discard some leading conv blobs (blobs are ordered from
            # max/coarsest level to min/finest level)
            self.num_roi_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1

            # Retain only the spatial scales that will be used for RoI heads. `Conv_Body.spatial_scale`
            # may include extra scales that are used for RPN proposals, but not for RoI heads.
            self.Conv_Body.spatial_scale = self.Conv_Body.spatial_scale[-self.num_roi_levels:]

        # BBOX Branch
        if not cfg.MODEL.RPN_ONLY:
            self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            self.Box_Outs = fast_rcnn_heads.fast_rcnn_outputs(
                self.Box_Head.dim_out)

        # Modules for EFPN
        if cfg.MODEL.SR:
            # Feature Texture Transfer Module
            self.Extension = Extension(2)
            # Finetuned detector head for the extended pyramid level
            self.Box_Head_SR = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            self.Box_Outs_SR = fast_rcnn_heads.fast_rcnn_outputs(
                self.Box_Head.dim_out)

        # Online Hard Example Mining
        if cfg.TRAIN.OHEM:
            self.ohem = fast_rcnn_heads.OHEM_Sampler()

        self._init_modules()

    def _init_modules(self):
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            resnet_utils.load_pretrained_imagenet_weights(self)
            # Check if shared weights are equaled
            if cfg.MODEL.MASK_ON and getattr(self.Mask_Head, 'SHARE_RES5', False):
                assert compare_state_dict(self.Mask_Head.res5.state_dict(), self.Box_Head.res5.state_dict())
            if cfg.MODEL.KEYPOINTS_ON and getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                assert compare_state_dict(self.Keypoint_Head.res5.state_dict(), self.Box_Head.res5.state_dict())

        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

        # ######TODO: freeze all branches except iou branch!!
        if cfg.TRAIN.SR_ONLY:
            for p in self.parameters(): p.requires_grad = False
            for p in self.Extension.parameters(): p.requires_grad = True
        if cfg.TRAIN.SR_FINETUNE:
            for p in self.parameters(): p.requires_grad = False
            if cfg.TRAIN.OHEM:
                for p in self.Extension.parameters(): p.requires_grad = True
            for p in self.Box_Head_SR.parameters(): p.requires_grad = True
            for p in self.Box_Outs_SR.parameters(): p.requires_grad = True

    def forward(self, data, im_info,  roidb=None,
                **rpn_kwargs):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, im_info,  roidb, **rpn_kwargs)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, im_info,  roidb, **rpn_kwargs)

    def _forward(self, data, im_info,  roidb=None,

                 **rpn_kwargs):

        im_data = data

        if self.training:
            roidb = list(map(lambda x: blob_utils.deserialize(x)[0], roidb))

        return_dict = {}  # A dict to collect return variables

        # Construction of top 4 layers of EFPN
        blob_conv, C1_2, fpn_inner2, fpn_inner3 = self.Conv_Body(im_data)  # blob_conv is a list

        if not self.training or not cfg.TRAIN.SR_ONLY:
            rpn_ret = self.RPN(blob_conv[0:5], im_info, roidb)


        if cfg.MODEL.SR:
            if cfg.TRAIN.SR_ONLY and self.training:
                m = nn.AvgPool2d(2, stride=2, padding=0)
                im_data_lr = m(im_data)
                blob_conv_lr, C1_2_lr, fpn_inner2_lr, fpn_inner3_lr = self.Conv_Body(im_data_lr)  # P6, P5, P4, P3, P2

                # Extension in EFPN
                P2_sr, fpn_inner3_sr = self.Extension(fpn_inner3_lr, fpn_inner2_lr, C1_2_lr)

                # Global reconstruction loss
                loss_P2_sr = compute_SR_loss(P2_sr, blob_conv[-1])
                loss_fpn_inner3 = compute_SR_loss(fpn_inner3_sr, fpn_inner3)

                # Positive Patch Loss
                gt_boxes_batches = np.zeros((0, 5))
                for batch_idx in range(len(roidb)):
                    gt_boxes = np.zeros((roidb[batch_idx]['boxes'].shape[0], 5))
                    gt_boxes[:, 0] = batch_idx
                    gt_boxes[:, 1:5] = roidb[batch_idx]['boxes'] * (im_info[batch_idx].data.numpy()[2])
                    gt_boxes_batches = np.concatenate((gt_boxes_batches, gt_boxes), axis=0)
                patchloss_P2_sr = compute_SR_patchloss(P2_sr, blob_conv[-1], gt_boxes_batches, sc=0.25)
                patchloss_fpn_inner3 = compute_SR_patchloss(fpn_inner3_sr, fpn_inner3, gt_boxes_batches, sc=0.25/2)

                # Foreground-Background-Balanced Loss
                k = 1
                return_dict['losses'] = {}
                return_dict['metrics'] = {}
                return_dict['losses']['loss_P2_sr'] = loss_P2_sr
                return_dict['losses']['loss_fpninner3_sr'] = loss_fpn_inner3 * k
                return_dict['losses']['loss_P2_srpatch'] = patchloss_P2_sr
                return_dict['losses']['loss_fpninner3_srpatch'] = patchloss_fpn_inner3 * k
                return return_dict
            else:
                # Texture Transfer Method
                P2_sr, fpn_inner3_sr = self.Extension(fpn_inner3, fpn_inner2, C1_2)
                blob_conv.append(P2_sr)

        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            ###TODO:!!!
            blob_conv = blob_conv[-(self.num_roi_levels + 1):]

        if not cfg.MODEL.RPN_ONLY:
            # Use original detector head
            if not cfg.TRAIN.SR_FINETUNE:
                if cfg.MODEL.SHARE_RES5 and self.training:
                    box_feat, res5_feat = self.Box_Head(blob_conv, rpn_ret)
                else:
                    box_feat = self.Box_Head(blob_conv, rpn_ret)
                cls_score, bbox_pred = self.Box_Outs(box_feat)

            # Combine a new finetuned detector head with the original head
            else:
                if self.training:
                    box_feat1 = self.Box_Head_SR(P2_sr, rpn_ret, sc=0.5)
                    cls_score1, bbox_pred1 = self.Box_Outs_SR(box_feat1)

                    num_fpn1 = len(rpn_ret['rois_fpn1'])
                    rois_idx_order = np.argsort(rpn_ret['rois_idx_restore_int32']).astype(np.int32, copy=False)
                    fpn1_idx = rois_idx_order[0:num_fpn1]

                    if cfg.TRAIN.OHEM:
                        ohem_inds = self.ohem(cls_score1, bbox_pred1, rpn_ret['labels_int32'][fpn1_idx],
                                              rpn_ret['bbox_targets'][fpn1_idx, :],
                                              rpn_ret['bbox_inside_weights'][fpn1_idx, :],
                                              rpn_ret['bbox_outside_weights'][fpn1_idx, :])
                        loss_cls, loss_bbox, accuracy_cls = fast_rcnn_heads.fast_rcnn_losses(
                            cls_score1[ohem_inds], bbox_pred1[ohem_inds],
                            rpn_ret['labels_int32'][fpn1_idx][ohem_inds],
                            rpn_ret['bbox_targets'][fpn1_idx, :][ohem_inds],
                            rpn_ret['bbox_inside_weights'][fpn1_idx, :][ohem_inds],
                            rpn_ret['bbox_outside_weights'][fpn1_idx, :][ohem_inds])
                    else:

                        loss_cls, loss_bbox, accuracy_cls = fast_rcnn_heads.fast_rcnn_losses(
                            cls_score1, bbox_pred1, rpn_ret['labels_int32'][fpn1_idx],
                            rpn_ret['bbox_targets'][fpn1_idx, :],
                            rpn_ret['bbox_inside_weights'][fpn1_idx, :], rpn_ret['bbox_outside_weights'][fpn1_idx, :])

                    return_dict['losses'] = {}
                    return_dict['metrics'] = {}
                    return_dict['losses']['loss_cls'] = loss_cls
                    return_dict['losses']['loss_bbox'] = loss_bbox
                    return_dict['metrics']['accuracy_cls'] = accuracy_cls
                    return return_dict
                else:
                    if len(rpn_ret['rois_fpn1']):
                        box_feat_SR = self.Box_Head_SR(P2_sr, rpn_ret, sc=0.5)
                        cls_score_SR1, bbox_pred_SR1 = self.Box_Outs_SR(box_feat_SR)

                        box_feat = self.Box_Head(blob_conv, rpn_ret)
                        cls_score, bbox_pred = self.Box_Outs(box_feat)
                        num_fpn1 = len(rpn_ret['rois_fpn1'])
                        rois_idx_order = np.argsort(rpn_ret['rois_idx_restore_int32']).astype(np.int32, copy=False)
                        fpn1_idx = rois_idx_order[0:num_fpn1]

                        cls_score[fpn1_idx, :] = cls_score_SR1
                        bbox_pred[fpn1_idx, :] = bbox_pred_SR1
                    else:
                        box_feat = self.Box_Head(blob_conv, rpn_ret)
                        cls_score, bbox_pred = self.Box_Outs(box_feat)


        if self.training:
            return_dict['losses'] = {}
            return_dict['metrics'] = {}
            # rpn loss
            rpn_kwargs.update(dict(
                (k, rpn_ret[k]) for k in rpn_ret.keys()
                if (k.startswith('rpn_cls_logits') or k.startswith('rpn_bbox_pred'))
            ))
            loss_rpn_cls, loss_rpn_bbox = rpn_heads.generic_rpn_losses(**rpn_kwargs)
            if cfg.FPN.FPN_ON:
                num_rpn_levels = len(loss_rpn_cls)
                for i, lvl in enumerate(range(cfg.FPN.RPN_MAX_LEVEL + 1 - num_rpn_levels, cfg.FPN.RPN_MAX_LEVEL + 1)):
                    return_dict['losses']['loss_rpn_cls_fpn%d' % lvl] = loss_rpn_cls[i]
                    return_dict['losses']['loss_rpn_bbox_fpn%d' % lvl] = loss_rpn_bbox[i]
            else:
                return_dict['losses']['loss_rpn_cls'] = loss_rpn_cls
                return_dict['losses']['loss_rpn_bbox'] = loss_rpn_bbox

            # bbox loss
            loss_cls, loss_bbox, accuracy_cls = fast_rcnn_heads.fast_rcnn_losses(
                cls_score, bbox_pred, rpn_ret['labels_int32'], rpn_ret['bbox_targets'],
                rpn_ret['bbox_inside_weights'], rpn_ret['bbox_outside_weights'])

            return_dict['losses']['loss_cls'] = loss_cls
            return_dict['losses']['loss_bbox'] = loss_bbox
            return_dict['metrics']['accuracy_cls'] = accuracy_cls

            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)
            for k, v in return_dict['metrics'].items():
                return_dict['metrics'][k] = v.unsqueeze(0)

        else:
            # Testing
            # return_dict['blob_conv'] = blob_conv
            return_dict['blob_conv'] = None
            rois_boxes = []
            rois_scores = []
            for key in rpn_ret.keys():
                if 'rpn_rois_fpn' in key:
                    rois_boxes.append(rpn_ret[key])
                    rois_scores.append(rpn_ret[key.replace('rpn_rois_fpn', 'rpn_rois_prob_fpn')])
            rois_with_scores = np.vstack(rois_boxes)
            rois_with_scores[:, 0] = np.vstack(rois_scores).squeeze()
            if cfg.TEST.TEST_RPN:
                return rois_with_scores
            return_dict['rois'] = rpn_ret['rois']
            return_dict['cls_score'] = cls_score
            return_dict['bbox_pred'] = bbox_pred

        return return_dict

    def roi_feature_transform(self, blobs_in, rpn_ret, blob_rois='rois', method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoICrop', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        if isinstance(blobs_in, list):
            # FPN case: add RoIFeatureTransform to each FPN level
            device_id = blobs_in[0].get_device()
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            ####TODO:!!!
            assert len(blobs_in) == k_max - k_min + 1 + 1
            bl_out_list = []
            ####TODO:!!!
            for lvl in range(k_min - 1, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
                #####TODO:!!!
                if lvl == k_min - 1:
                    sc = 0.25 * 2
                else:
                    sc = spatial_scale[k_max - lvl]  # in reversed order
                #####
                bl_rois = blob_rois + '_fpn' + str(lvl)
                if len(rpn_ret[bl_rois]):
                    rois = Variable(torch.from_numpy(rpn_ret[bl_rois]).float()).cuda(device_id)
                    if method == 'RoIPoolF':
                        # Warning!: Not check if implementation matches Detectron
                        xform_out = RoIPoolFunction(resolution, resolution, sc)(bl_in, rois)
                    elif method == 'RoICrop':
                        # Warning!: Not check if implementation matches Detectron
                        grid_xy = net_utils.affine_grid_gen(
                            rois, bl_in.size()[2:], self.grid_size)
                        grid_yx = torch.stack(
                            [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                        xform_out = RoICropFunction()(bl_in, Variable(grid_yx).detach())
                        if cfg.CROP_RESIZE_WITH_MAX_POOL:
                            xform_out = F.max_pool2d(xform_out, 2, 2)
                    elif method == 'RoIAlign':
                        xform_out = RoIAlignFunction(
                            resolution, resolution, sc, sampling_ratio)(bl_in, rois)
                    bl_out_list.append(xform_out)

            # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
            xform_shuffled = torch.cat(bl_out_list, dim=0)

            # Unshuffle to match rois from dataloader
            device_id = xform_shuffled.get_device()
            restore_bl = rpn_ret[blob_rois + '_idx_restore_int32']
            restore_bl = Variable(
                torch.from_numpy(restore_bl.astype('int64', copy=False))).cuda(device_id)
            xform_out = xform_shuffled[restore_bl]
        else:
            # Single feature level
            # rois: holds R regions of interest, each is a 5-tuple
            # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
            # rectangle (x1, y1, x2, y2)
            device_id = blobs_in.get_device()
            # rois = Variable(torch.from_numpy(rpn_ret[blob_rois]).float()).cuda(device_id)
            ###TODO:!!!
            rois = Variable(torch.from_numpy(rpn_ret['rois_fpn1']).float()).cuda(device_id)
            if method == 'RoIPoolF':
                xform_out = RoIPoolFunction(resolution, resolution, spatial_scale)(blobs_in, rois)
            elif method == 'RoICrop':
                grid_xy = net_utils.affine_grid_gen(rois, blobs_in.size()[2:], self.grid_size)
                grid_yx = torch.stack(
                    [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                xform_out = RoICropFunction()(blobs_in, Variable(grid_yx).detach())
                if cfg.CROP_RESIZE_WITH_MAX_POOL:
                    xform_out = F.max_pool2d(xform_out, 2, 2)
            elif method == 'RoIAlign':
                xform_out = RoIAlignFunction(
                    resolution, resolution, spatial_scale, sampling_ratio)(blobs_in, rois)

        return xform_out

    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]
        return blob_conv

    @check_inference
    def mask_net(self, blob_conv, rpn_blob):
        """For inference"""
        mask_feat = self.Mask_Head(blob_conv, rpn_blob)
        mask_pred = self.Mask_Outs(mask_feat)
        return mask_pred

    @check_inference
    def keypoint_net(self, blob_conv, rpn_blob):
        """For inference"""
        kps_feat = self.Keypoint_Head(blob_conv, rpn_blob)
        kps_pred = self.Keypoint_Outs(kps_feat)
        return kps_pred

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value

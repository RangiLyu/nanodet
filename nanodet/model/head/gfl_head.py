import math

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from nanodet.util import (
    bbox2distance,
    distance2bbox,
    images_to_levels,
    multi_apply,
    overlay_bbox_cv,
)

from ...data.transform.warp import warp_boxes
from ..loss.gfocal_loss import DistributionFocalLoss, QualityFocalLoss
from ..loss.iou_loss import GIoULoss, bbox_overlaps
from ..module.conv import ConvModule
from ..module.init_weights import normal_init
from ..module.nms import multiclass_nms
from ..module.scale import Scale
from .assigner.atss_assigner import ATSSAssigner


def reduce_mean(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.true_divide(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}
    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer(
            "project", torch.linspace(0, self.reg_max, self.reg_max + 1)
        )

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.
        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        shape = x.size()
        x = F.softmax(x.reshape(*shape[:-1], 4, self.reg_max + 1), dim=-1)
        x = F.linear(x, self.project.type_as(x)).reshape(*shape[:-1], 4)
        return x


class GFLHead(nn.Module):
    """Generalized Focal Loss: Learning Qualified and Distributed Bounding
    Boxes for Dense Object Detection.

    GFL head structure is similar with ATSS, however GFL uses
    1) joint representation for classification and localization quality, and
    2) flexible General distribution for bounding box locations,
    which are supervised by
    Quality Focal Loss (QFL) and Distribution Focal Loss (DFL), respectively

    https://arxiv.org/abs/2006.04388

    :param num_classes: Number of categories excluding the background category.
    :param loss: Config of all loss functions.
    :param input_channel: Number of channels in the input feature map.
    :param feat_channels: Number of conv layers in cls and reg tower. Default: 4.
    :param stacked_convs: Number of conv layers in cls and reg tower. Default: 4.
    :param octave_base_scale: Scale factor of grid cells.
    :param strides: Down sample strides of all level feature map
    :param conv_cfg: Dictionary to construct and config conv layer. Default: None.
    :param norm_cfg: Dictionary to construct and config norm layer.
    :param reg_max: Max value of integral set :math: `{0, ..., reg_max}`
                    in QFL setting. Default: 16.
    :param kwargs:
    """

    def __init__(
        self,
        num_classes,
        loss,
        input_channel,
        feat_channels=256,
        stacked_convs=4,
        octave_base_scale=4,
        strides=[8, 16, 32],
        conv_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
        reg_max=16,
        ignore_iof_thr=-1,
        **kwargs
    ):
        super(GFLHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = input_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.grid_cell_scale = octave_base_scale
        self.strides = strides
        self.reg_max = reg_max

        self.loss_cfg = loss
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_sigmoid = self.loss_cfg.loss_qfl.use_sigmoid
        self.ignore_iof_thr = ignore_iof_thr
        if self.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.assigner = ATSSAssigner(topk=9, ignore_iof_thr=ignore_iof_thr)
        self.distribution_project = Integral(self.reg_max)

        self.loss_qfl = QualityFocalLoss(
            use_sigmoid=self.use_sigmoid,
            beta=self.loss_cfg.loss_qfl.beta,
            loss_weight=self.loss_cfg.loss_qfl.loss_weight,
        )
        self.loss_dfl = DistributionFocalLoss(
            loss_weight=self.loss_cfg.loss_dfl.loss_weight
        )
        self.loss_bbox = GIoULoss(loss_weight=self.loss_cfg.loss_bbox.loss_weight)
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                )
            )
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                )
            )
        self.gfl_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1
        )
        self.gfl_reg = nn.Conv2d(
            self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1
        )
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = -4.595
        normal_init(self.gfl_cls, std=0.01, bias=bias_cls)
        normal_init(self.gfl_reg, std=0.01)

    def forward(self, feats):
        if torch.onnx.is_in_onnx_export():
            return self._forward_onnx(feats)
        outputs = []
        for x, scale in zip(feats, self.scales):
            cls_feat = x
            reg_feat = x
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)
            cls_score = self.gfl_cls(cls_feat)
            bbox_pred = scale(self.gfl_reg(reg_feat)).float()
            output = torch.cat([cls_score, bbox_pred], dim=1)
            outputs.append(output.flatten(start_dim=2))
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
        return outputs

    def loss(self, preds, gt_meta):
        cls_scores, bbox_preds = preds.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )
        device = cls_scores.device
        gt_bboxes = gt_meta["gt_bboxes"]
        gt_bboxes_ignore = gt_meta["gt_bboxes_ignore"]
        gt_labels = gt_meta["gt_labels"]
        input_height, input_width = gt_meta["img"].shape[2:]

        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width) / stride)
            for stride in self.strides
        ]

        cls_reg_targets = self.target_assign(
            cls_scores,
            bbox_preds,
            featmap_sizes,
            gt_bboxes,
            gt_bboxes_ignore,
            gt_labels,
            device=device,
        )
        if cls_reg_targets is None:
            return None

        (
            cls_preds_list,
            reg_preds_list,
            grid_cells_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets

        num_total_samples = reduce_mean(torch.tensor(num_total_pos).to(device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_qfl, losses_bbox, losses_dfl, avg_factor = multi_apply(
            self.loss_single,
            grid_cells_list,
            cls_preds_list,
            reg_preds_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            self.strides,
            num_total_samples=num_total_samples,
        )

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).item()
        if avg_factor <= 0:
            loss_qfl = torch.tensor(0, dtype=torch.float32, requires_grad=True).to(
                device
            )
            loss_bbox = torch.tensor(0, dtype=torch.float32, requires_grad=True).to(
                device
            )
            loss_dfl = torch.tensor(0, dtype=torch.float32, requires_grad=True).to(
                device
            )
        else:
            losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
            losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))

            loss_qfl = sum(losses_qfl)
            loss_bbox = sum(losses_bbox)
            loss_dfl = sum(losses_dfl)

        loss = loss_qfl + loss_bbox + loss_dfl
        loss_states = dict(loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl)

        return loss, loss_states

    def loss_single(
        self,
        grid_cells,
        cls_score,
        bbox_pred,
        labels,
        label_weights,
        bbox_targets,
        stride,
        num_total_samples,
    ):
        grid_cells = grid_cells.reshape(-1, 4)
        cls_score = cls_score.reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.reshape(-1, 4 * (self.reg_max + 1))
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = torch.nonzero(
            (labels >= 0) & (labels < bg_class_ind), as_tuple=False
        ).squeeze(1)

        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]  # (n, 4 * (reg_max + 1))
            pos_grid_cells = grid_cells[pos_inds]
            pos_grid_cell_centers = self.grid_cells_to_center(pos_grid_cells) / stride

            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.distribution_project(pos_bbox_pred)
            pos_decode_bbox_pred = distance2bbox(
                pos_grid_cell_centers, pos_bbox_pred_corners
            )
            pos_decode_bbox_targets = pos_bbox_targets / stride
            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(), pos_decode_bbox_targets, is_aligned=True
            )
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = bbox2distance(
                pos_grid_cell_centers, pos_decode_bbox_targets, self.reg_max
            ).reshape(-1)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0,
            )

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0,
            )
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = torch.tensor(0).to(cls_score.device)

        # qfl loss
        loss_qfl = self.loss_qfl(
            cls_score,
            (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples,
        )

        return loss_qfl, loss_bbox, loss_dfl, weight_targets.sum()

    def target_assign(
        self,
        cls_preds,
        reg_preds,
        featmap_sizes,
        gt_bboxes_list,
        gt_bboxes_ignore_list,
        gt_labels_list,
        device,
    ):
        """
        Assign target for a batch of images.
        :param batch_size: num of images in one batch
        :param featmap_sizes: A list of all grid cell boxes in all image
        :param gt_bboxes_list: A list of ground truth boxes in all image
        :param gt_bboxes_ignore_list: A list of all ignored boxes in all image
        :param gt_labels_list: A list of all ground truth label in all image
        :param device: pytorch device
        :return: Assign results of all images.
        """
        batch_size = cls_preds.shape[0]
        # get grid cells of one image
        multi_level_grid_cells = [
            self.get_grid_cells(
                featmap_sizes[i],
                self.grid_cell_scale,
                stride,
                dtype=torch.float32,
                device=device,
            )
            for i, stride in enumerate(self.strides)
        ]
        mlvl_grid_cells_list = [multi_level_grid_cells for i in range(batch_size)]

        # pixel cell number of multi-level feature maps
        num_level_cells = [grid_cells.size(0) for grid_cells in mlvl_grid_cells_list[0]]
        num_level_cells_list = [num_level_cells] * batch_size
        # concat all level cells and to a single tensor
        for i in range(batch_size):
            mlvl_grid_cells_list[i] = torch.cat(mlvl_grid_cells_list[i])
        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(batch_size)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(batch_size)]
        # target assign on all images, get list of tensors
        # list length = batch size
        # tensor first dim = num of all grid cell
        (
            all_grid_cells,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self.target_assign_single_img,
            mlvl_grid_cells_list,
            num_level_cells_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
        )
        # no valid cells
        if any([labels is None for labels in all_labels]):
            return None
        # sampled cells of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # merge list of targets tensors into one batch then split to multi levels
        mlvl_cls_preds = images_to_levels([c for c in cls_preds], num_level_cells)
        mlvl_reg_preds = images_to_levels([r for r in reg_preds], num_level_cells)
        mlvl_grid_cells = images_to_levels(all_grid_cells, num_level_cells)
        mlvl_labels = images_to_levels(all_labels, num_level_cells)
        mlvl_label_weights = images_to_levels(all_label_weights, num_level_cells)
        mlvl_bbox_targets = images_to_levels(all_bbox_targets, num_level_cells)
        mlvl_bbox_weights = images_to_levels(all_bbox_weights, num_level_cells)
        return (
            mlvl_cls_preds,
            mlvl_reg_preds,
            mlvl_grid_cells,
            mlvl_labels,
            mlvl_label_weights,
            mlvl_bbox_targets,
            mlvl_bbox_weights,
            num_total_pos,
            num_total_neg,
        )

    def target_assign_single_img(
        self, grid_cells, num_level_cells, gt_bboxes, gt_bboxes_ignore, gt_labels
    ):
        """
        Using ATSS Assigner to assign target on one image.
        :param grid_cells: Grid cell boxes of all pixels on feature map
        :param num_level_cells: numbers of grid cells on each level's feature map
        :param gt_bboxes: Ground truth boxes
        :param gt_bboxes_ignore: Ground truths which are ignored
        :param gt_labels: Ground truth labels
        :return: Assign results of a single image
        """
        device = grid_cells.device
        gt_bboxes = torch.from_numpy(gt_bboxes).to(device)
        gt_labels = torch.from_numpy(gt_labels).to(device)

        if gt_bboxes_ignore is not None:
            gt_bboxes_ignore = torch.from_numpy(gt_bboxes_ignore).to(device)

        assign_result = self.assigner.assign(
            grid_cells, num_level_cells, gt_bboxes, gt_bboxes_ignore, gt_labels
        )

        pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds = self.sample(
            assign_result, gt_bboxes
        )

        num_cells = grid_cells.shape[0]
        bbox_targets = torch.zeros_like(grid_cells)
        bbox_weights = torch.zeros_like(grid_cells)
        labels = grid_cells.new_full((num_cells,), self.num_classes, dtype=torch.long)
        label_weights = grid_cells.new_zeros(num_cells, dtype=torch.float)

        if len(pos_inds) > 0:
            pos_bbox_targets = pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[pos_assigned_gt_inds]

            label_weights[pos_inds] = 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        return (
            grid_cells,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            pos_inds,
            neg_inds,
        )

    def sample(self, assign_result, gt_bboxes):
        pos_inds = (
            torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        neg_inds = (
            torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds

    def post_process(self, preds, meta):
        cls_scores, bbox_preds = preds.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )
        result_list = self.get_bboxes(cls_scores, bbox_preds, meta)
        det_results = {}
        warp_matrixes = (
            meta["warp_matrix"]
            if isinstance(meta["warp_matrix"], list)
            else meta["warp_matrix"]
        )
        img_heights = (
            meta["img_info"]["height"].cpu().numpy()
            if isinstance(meta["img_info"]["height"], torch.Tensor)
            else meta["img_info"]["height"]
        )
        img_widths = (
            meta["img_info"]["width"].cpu().numpy()
            if isinstance(meta["img_info"]["width"], torch.Tensor)
            else meta["img_info"]["width"]
        )
        img_ids = (
            meta["img_info"]["id"].cpu().numpy()
            if isinstance(meta["img_info"]["id"], torch.Tensor)
            else meta["img_info"]["id"]
        )

        for result, img_width, img_height, img_id, warp_matrix in zip(
            result_list, img_widths, img_heights, img_ids, warp_matrixes
        ):
            det_result = {}
            det_bboxes, det_labels = result
            det_bboxes = det_bboxes.detach().cpu().numpy()
            det_bboxes[:, :4] = warp_boxes(
                det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height
            )
            classes = det_labels.detach().cpu().numpy()
            for i in range(self.num_classes):
                inds = classes == i
                det_result[i] = np.concatenate(
                    [
                        det_bboxes[inds, :4].astype(np.float32),
                        det_bboxes[inds, 4:5].astype(np.float32),
                    ],
                    axis=1,
                ).tolist()
            det_results[img_id] = det_result
        return det_results

    def show_result(
        self, img, dets, class_names, score_thres=0.3, show=True, save_path=None
    ):
        result = overlay_bbox_cv(img, dets, class_names, score_thresh=score_thres)
        if show:
            cv2.imshow("det", result)
        return result

    def get_bboxes(self, cls_preds, reg_preds, img_metas):
        """Decode the outputs to bboxes.
        Args:
            cls_preds (Tensor): Shape (num_imgs, num_points, num_classes).
            reg_preds (Tensor): Shape (num_imgs, num_points, 4 * (regmax + 1)).
            img_metas (dict): Dict of image info.

        Returns:
            results_list (list[tuple]): List of detection bboxes and labels.
        """
        device = cls_preds.device
        b = cls_preds.shape[0]
        input_height, input_width = img_metas["img"].shape[2:]
        input_shape = (input_height, input_width)

        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width) / stride)
            for stride in self.strides
        ]
        # get grid cells of one image
        mlvl_center_priors = []
        for i, stride in enumerate(self.strides):
            y, x = self.get_single_level_center_point(
                featmap_sizes[i], stride, torch.float32, device
            )
            strides = x.new_full((x.shape[0],), stride)
            proiors = torch.stack([x, y, strides, strides], dim=-1)
            mlvl_center_priors.append(proiors.unsqueeze(0).repeat(b, 1, 1))

        center_priors = torch.cat(mlvl_center_priors, dim=1)
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
        scores = cls_preds.sigmoid()
        result_list = []
        for i in range(b):
            # add a dummy background class at the end of all labels
            # same with mmdetection2.0
            score, bbox = scores[i], bboxes[i]
            padding = score.new_zeros(score.shape[0], 1)
            score = torch.cat([score, padding], dim=1)
            results = multiclass_nms(
                bbox,
                score,
                score_thr=0.05,
                nms_cfg=dict(type="nms", iou_threshold=0.6),
                max_num=100,
            )
            result_list.append(results)
        return result_list

    def get_single_level_center_point(
        self, featmap_size, stride, dtype, device, flatten=True
    ):
        """
        Generate pixel centers of a single stage feature map.
        :param featmap_size: height and width of the feature map
        :param stride: down sample stride of the feature map
        :param dtype: data type of the tensors
        :param device: device of the tensors
        :param flatten: flatten the x and y tensors
        :return: y and x of the center points
        """
        h, w = featmap_size
        x_range = (torch.arange(w, dtype=dtype, device=device) + 0.5) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device) + 0.5) * stride
        y, x = torch.meshgrid(y_range, x_range)
        if flatten:
            y = y.flatten()
            x = x.flatten()
        return y, x

    def get_grid_cells(self, featmap_size, scale, stride, dtype, device):
        """
        Generate grid cells of a feature map for target assignment.
        :param featmap_size: Size of a single level feature map.
        :param scale: Grid cell scale.
        :param stride: Down sample stride of the feature map.
        :param dtype: Data type of the tensors.
        :param device: Device of the tensors.
        :return: Grid_cells xyxy position. Size should be [feat_w * feat_h, 4]
        """
        cell_size = stride * scale
        y, x = self.get_single_level_center_point(
            featmap_size, stride, dtype, device, flatten=True
        )
        grid_cells = torch.stack(
            [
                x - 0.5 * cell_size,
                y - 0.5 * cell_size,
                x + 0.5 * cell_size,
                y + 0.5 * cell_size,
            ],
            dim=-1,
        )
        return grid_cells

    def grid_cells_to_center(self, grid_cells):
        """
        Get center location of each gird cell
        :param grid_cells: grid cells of a feature map
        :return: center points
        """
        cells_cx = (grid_cells[:, 2] + grid_cells[:, 0]) / 2
        cells_cy = (grid_cells[:, 3] + grid_cells[:, 1]) / 2
        return torch.stack([cells_cx, cells_cy], dim=-1)

    def _forward_onnx(self, feats):
        """only used for onnx export"""
        outputs = []
        for x, scale in zip(feats, self.scales):
            cls_feat = x
            reg_feat = x
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)
            cls_pred = self.gfl_cls(cls_feat)
            reg_pred = scale(self.gfl_reg(reg_feat))
            cls_pred = cls_pred.sigmoid()
            out = torch.cat([cls_pred, reg_pred], dim=1)
            outputs.append(out.flatten(start_dim=2))
        return torch.cat(outputs, dim=2).permute(0, 2, 1)

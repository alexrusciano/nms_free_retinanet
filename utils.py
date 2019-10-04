from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def to_cpu(tensor):
    return tensor.detach().cpu()


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def broadcast_bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    (IMAGES, PRED, 4) with (IMAGES, TRUE, 4) makes (IMAGES, PRED, TRUE)
    """
    with torch.no_grad():
        # in order to broadcast, need to append dimensions to box
        box1 = box1.unsqueeze(-1)
        box2 = box2.unsqueeze(1)
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,:, 0,:], box1[:,:, 1,:], box1[:,:, 2,:], box1[:,:, 3,:]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,:,:, 0], box2[:,:,:, 1], box2[:,:,:, 2], box2[:,:,:, 3]
        # get the corrdinates of the intersection rectangle
        # these are going to be (IMAGES, PRED, TRUE)
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
        # Intersection area
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1 + 1, min=0)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def nms_dup(predictions, scores, label, dup_scores,nms_thresh=.5):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    detections = to_cpu(predictions)
    scores = to_cpu(scores)
    label = to_cpu(label)
    used_status = torch.round(to_cpu(dup_scores))
    indices_perm = torch.arange(0, label.shape[0])
    # Filter out confidence scores below threshold
    # If none are remaining => process next image
    # Object confidence times class confidence
    # Sort by it
    # Perform non-maximum suppression
    keep_boxes = []
    keep_indices = torch.ByteTensor(label.shape).fill_(0)
    while detections.shape[0]:
        large_overlap = bbox_iou(detections[0].unsqueeze(0), detections[:, :4]) > nms_thresh
        label_match = label[0] == label
        # Indices of boxes with lower confidence scores, large IOUs and matching labels
        invalid = large_overlap & label_match
        #however we have to be forgiving to those with a good duplicate score
        forgiven = invalid & (used_status > .5)
        used_status = used_status - forgiven.float()
        invalid = invalid  & (~ forgiven)

        keep_indices[indices_perm[0]]=torch.tensor(True)

        used_status = used_status[~invalid]
        detections = detections[ ~ invalid]
        label = label[~invalid]
        scores=scores[~ invalid]
        indices_perm = indices_perm[~ invalid]
        #dup_scores = dup_scores[~ invalid]

    return keep_indices


def inter_anchor_nms(predictions, scores, label, source, nms_thresh=.5):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    image_pred = to_cpu(predictions)
    scores = to_cpu(scores)
    label = to_cpu(label)
    source = to_cpu(source)
    output = None
    # Filter out confidence scores below threshold
    # If none are remaining => process next image
    # Object confidence times class confidence
    # Sort by it
    indices_perm  = (-scores).argsort()
    detections = image_pred[indices_perm,:]
    label = label[indices_perm]
    scores = scores[indices_perm]
    source = source[indices_perm]
    nms_thresh=.5
    # Perform non-maximum suppression
    keep_boxes = []
    keep_indices = torch.ByteTensor(label.shape).fill_(0)
    while detections.shape[0]:
        large_overlap = bbox_iou(detections[0].unsqueeze(0), detections[:, :4]) > nms_thresh
        label_match = label[0] == label
        anchor_mismatch = source[0] != source
        #needs to remove itself after all
        anchor_mismatch[0]= ~ anchor_mismatch[0]
        # Indices of boxes with lower confidence scores, large IOUs and matching labels
        invalid = large_overlap & label_match & anchor_mismatch
        weights = scores[invalid]
        keep_indices[indices_perm[0]]=torch.tensor(True)
        detections = detections[ ~ invalid]
        label = label[~invalid]
        scores=scores[~ invalid]
        indices_perm = indices_perm[~ invalid]
        source = source[~invalid]
    if keep_boxes:
        output = torch.stack(keep_boxes)

    return keep_indices


def non_max_suppression(predictions, scores, label, nms_thresh=.5):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    image_pred = to_cpu(predictions)
    scores = to_cpu(scores)
    label = to_cpu(label)
    output = None
    # Filter out confidence scores below threshold
    # If none are remaining => process next image
    # Object confidence times class confidence
    # Sort by it
    indices_perm  = (-scores).argsort()
    detections = image_pred[indices_perm,:]
    label = label[indices_perm]
    scores = scores[indices_perm]
    # Perform non-maximum suppression
    keep_boxes = []
    keep_indices = torch.ByteTensor(label.shape).fill_(0)
    while detections.shape[0]:
        large_overlap = bbox_iou(detections[0].unsqueeze(0), detections[:, :4]) > nms_thresh
        label_match = label[0] == label
        # Indices of boxes with lower confidence scores, large IOUs and matching labels
        invalid = large_overlap & label_match
        weights = scores[invalid]
        keep_indices[indices_perm[0]]=torch.tensor(True)
        detections = detections[ ~ invalid]
        label = label[~invalid]
        scores=scores[~ invalid]
        indices_perm = indices_perm[~ invalid]
    if keep_boxes:
        output = torch.stack(keep_boxes)

    return keep_indices

def bb_nms(predictions, scores, label, nms_thresh=None):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    if nms_thresh is None:
        nms_thresh = torch.ones(predictions.shape[0])*.5
    image_pred = to_cpu(predictions)
    scores = to_cpu(scores)
    label = to_cpu(label)
    nms_thresh= to_cpu(nms_thresh)
    output = None
    # Filter out confidence scores below threshold
    # If none are remaining => process next image
    # Object confidence times class confidence
    # Sort by it
    indices_perm  = (-scores).argsort()
    detections = image_pred[indices_perm,:]
    label = label[indices_perm]
    scores = scores[indices_perm]
    nms_thresh = nms_thresh[indices_perm]
    # Perform non-maximum suppression
    keep_boxes = []
    keep_indices = torch.ByteTensor(label.shape).fill_(0)
    while detections.shape[0]:
        large_overlap = bbox_iou(detections[0].unsqueeze(0), detections[:, :4]) > nms_thresh
        label_match = label[0] == label
        # Indices of boxes with lower confidence scores, large IOUs and matching labels
        invalid = large_overlap & label_match
        weights = scores[invalid]
        # Merge overlapping bboxes by order of confidence
        detections[0, :4] = (weights * detections[invalid,:]).sum(dim=0) / weights.sum()
        keep_boxes += [detections[0]]
        keep_indices[indices_perm[0]]=torch.tensor(True)
        detections = detections[ ~ invalid]
        label = label[~invalid]
        scores=scores[~ invalid]
        indices_perm = indices_perm[~ invalid]
        nms_thresh = nms_thresh[~ invalid]
    if keep_boxes:
        output = torch.stack(keep_boxes)

    return keep_indices

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
        else:
            self.mean = mean
        #self.register_buffer('box_mean', self.mean)
        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
        else:
            self.std = std
        #self.register_buffer('box_stde',self.std)

    def forward(self, boxes, deltas):
        self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
        self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()

        widths  = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x   = boxes[:, :, 0] + 0.5 * widths
        ctr_y   = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w     = torch.exp(dw) * widths
        pred_h     = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):

        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes

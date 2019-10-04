import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from utils import bbox_iou, broadcast_bbox_iou

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

class FocalLoss(nn.Module):
    #def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().cuda())
                classification_losses.append(torch.tensor(0).float().cuda())

                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            #import pdb
            #pdb.set_trace()

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1
            targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones(targets.shape).cuda() * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()


                negative_indices = 1 - positive_indices
                regression_diff = torch.abs(targets - regression[positive_indices, :])
                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).float().cuda())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)

def _get_batch_statistics(iou_info, anchor_info, pred_labels, true_labels, assignments, sample_i,iou_thresh,iou_thresh_dup):
    '''
    Computes the target assignments for sample_i. assignments[sample_i] is (pred,).
        1 2 3
        4 5 6, 0 for being NOT a duplicate, 10 for from finer layer, 11 for coarser layer 
        7 8 9
    '''
    sample_ious = iou_info[sample_i]
    sample_anchor = anchor_info[sample_i]
    sample_pred_labels = pred_labels[sample_i]
    sample_true_labels = true_labels[sample_i]
    
    # we create one list for each possible object category
    all_detected_boxes = [{} for j in range(100)]
    num_annotations = (sample_true_labels > -1).sum()
    found = 0
    dups = 0
    for pred_i, label in enumerate(sample_pred_labels):
        # for each pred_i, we only consider the ground truth of the same label
        usable_ious = sample_true_labels == label
        total = usable_ious.sum()
        if total==0:
            continue
        detected_boxes = all_detected_boxes[label]
        eligible_ious = sample_ious[pred_i, usable_ious]
        iou, box_index = torch.max(eligible_ious,dim=0)
        if iou >= iou_thresh and box_index.item() not in detected_boxes:
            assignments[sample_i, pred_i] = 0
            detected_boxes[box_index.item()] = pred_i
            found += 1
        
        # currently it is limiting the ratio
        elif iou >= iou_thresh_dup and box_index.item() in detected_boxes and dups < 11* found:
            x,y,l = sample_anchor[detected_boxes[box_index.item()]]-sample_anchor[pred_i]
            if l>0:
                assignments[sample_i,pred_i]=10
            elif l<0:
                # level 0 is finest, so negative means candidate object is from coarser level
                assignments[sample_i,pred_i]=11
            elif x < 0 and y < 0:
                assignments[sample_i,pred_i]= 1
            elif x < 0 and y == 0:
                assignments[sample_i,pred_i]= 2
            elif x < 0 and y > 0:
                assignments[sample_i,pred_i]=3
            elif x ==0 and y < 0:
                assignments[sample_i,pred_i]=4
            elif x ==0 and y ==0:
                assignments[sample_i,pred_i]=5
            elif x==0 and y > 0:
                assignments[sample_i,pred_i]=6
            elif x > 0 and y < 0:
                assignments[sample_i,pred_i]=7
            elif x > 0 and y == 0:
                assignments[sample_i,pred_i]=8
            elif x > 0 and y > 0:
                assignments[sample_i,pred_i]=9
            dups += 1
        if found == num_annotations and dups >= 11* found:
            break
    #print(num_annotations, found)


def DuplicateLoss1(labels_ot, duplicates_ot, iou_info, anchor_info, annotations):
    '''
    build the optimal duplicate assignments with _get_batch_statistics,
    apply cross-entropy loss to compare with model's duplicate predictions
    '''
    #no grad, we are building targets for duplicate loss
    with torch.no_grad():
        num_images = annotations.shape[0]
        
        model_labels = labels_ot.clone().cpu()
        true_labels = annotations[:,:,4].long()

        assignments = torch.zeros(model_labels.shape, dtype = torch.int8) -1

        for sample_i in range(num_images):
            #basically go through mAP calculation, recording the "true" objects
            _get_batch_statistics(iou_info, anchor_info, model_labels, true_labels, assignments, sample_i, .5,.5)

    assignments = assignments.cuda()
    loss = torch.tensor(0.0, requires_grad = True).cuda()

    for sample_i in range(num_images):
        sample_assignments = assignments[sample_i]
        sample_duplicates = duplicates_ot[sample_i, sample_assignments>-1,:]
        why = F.softmax(sample_duplicates)[:,0]
        #why = F.softmax(duplicates_ot[sample_i, sample_assignments==-1,:])[:100,0]
        #print(why)
        sample_assignments = sample_assignments[sample_assignments > -1].long()
        num_considered = sample_assignments.shape[0]
        #print("How many?", true_labels[sample_i,true_labels[sample_i]>-1].shape[0])
        #exit(0)
        outputs = F.log_softmax(sample_duplicates,dim=1)
        if outputs.shape[0] ==0:
            continue
        
        for j in range(12):
            saj = sample_assignments == j
            if saj.sum():
                outputs_j = outputs[saj,j]
                #print("onto" , j)
                #print(F.softmax(sample_duplicates[saj]))
                loss = loss - outputs_j.sum()/num_considered
                #print(loss)
    #exit(0)
    return loss / num_images

def DuplicateLoss2(labels_ot, duplicates_ot, iou_info, anchor_info, annotations):
        with torch.no_grad():
            num_images = annotations.shape[0]
        
            model_labels = labels_ot.clone().cpu()
            true_labels = annotations[:,:,4].long()

            assignments = torch.zeros(model_labels.shape, dtype = torch.int8) -1

            for sample_i in range(num_images):
                #basically go through mAP calculation, recording the "true" objects
                _get_batch_statistics(iou_info, anchor_info, model_labels, true_labels, assignments, sample_i, .6,.4)

        assignments = assignments.cuda()
        loss = torch.tensor(0.0, requires_grad = True).cuda()
    
        for sample_i in range(num_images):
            sample_assignments = assignments[sample_i]
            sample_duplicates = duplicates_ot[sample_i, sample_assignments>-1,:]
            sample_assignments = sample_assignments[sample_assignments > -1].long()
            num_considered = sample_assignments.shape[0]
            #tmp = sample_duplicates[:,0]+torch.
            outputs = F.sigmoid(sample_duplicates[:,0])
            if outputs.shape[0] ==0:
                continue
        
            sa0 = sample_assignments == 0
            if sa0.sum():
                outputs_0 = outputs[sa0]
                outputs_0 = outputs_0[outputs_0<.95]
                loss = loss - 10*torch.log(outputs_0).sum()/num_considered
                #outputs_0 = .8-outputs[sa0]
                #loss = loss + 5*outputs_0[outputs_0>0].sum()/num_considered

            sa1 = sample_assignments > 0
            if sa1.sum():
                outputs_1 = outputs[sa1]
                outputs_1 = outputs_1[outputs_1 > .05]
                loss = loss - torch.log(1-outputs_1).sum()/num_considered
                #outputs_1 = outputs[sa1] -.2
                #loss = loss + outputs_1[outputs_1>0].sum()/num_considered
        return loss / num_images
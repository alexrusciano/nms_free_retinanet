import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse
import model
import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
        parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
        parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)

        parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
        parser.add_argument('--coco_path', help='Path to COCO directory')
        parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
        parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
        parser.add_argument('--version', help ='new thing', type=int, default = 1)
        parser.add_argument('--model', help='Path to model (.pt) file.')
        parser.add_argument('--vis_dir', help = 'where to save files', type = str)
        parser = parser.parse_args(args)

        if parser.dataset == 'coco':
                dataset_val = CocoDataset(parser.coco_path, set_name='reduced', transform=transforms.Compose([Normalizer(), Resizer()]))
        elif parser.dataset == 'csv':
                dataset_val = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
        else:
                raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

        if parser.depth == 18:
                retinanet = model.resnet18(num_classes=dataset_val.num_classes(), pretrained=True)
        elif parser.depth == 34:
                retinanet = model.resnet34(num_classes=dataset_val.num_classes(), pretrained=True)
        elif parser.depth == 50 and parser.version == 0:
                print("old\n")
                retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
        elif parser.depth == 50 and parser.version == 1:
                print("new\n")
                retinanet = model.resnet50_op(num_classes=dataset_val.num_classes(), pretrained=True, duplicate_version = 0)

        elif parser.depth == 101:
                retinanet = model.resnet101(num_classes=dataset_val.num_classes(), pretrained=True)
        elif parser.depth == 152:
                retinanet = model.resnet152(num_classes=dataset_val.num_classes(), pretrained=True)
        else:
                raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

        retinanet.load_state_dict(torch.load(parser.model),strict=True)
        use_gpu = True

        if use_gpu:
                retinanet = retinanet.cuda()

        retinanet.eval()

        unnormalize = UnNormalizer()

        def draw_caption(image, box, caption):

                b = np.array(box).astype(int)
                cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
                cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        #version with nms
        count = 0
        for idx, data in enumerate(dataloader_val):

                with torch.no_grad():
                        st = time.time()
                        scores, classification, transformed_anchors, anchor_source = retinanet(data['img'].cuda().float())
                        scores = scores.cpu()
                        classification = classification.cpu()
                        print('Elapsed time: {}'.format(time.time()-st))
                        idxs = np.where(scores>0.5)
                        img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

                        img[img<0] = 0
                        img[img>255] = 255

                        img = np.transpose(img, (1, 2, 0))

                        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

                        for j in range(idxs[0].shape[0]):
                                bbox = transformed_anchors[idxs[0][j], :]
                                x1 = int(bbox[0])
                                y1 = int(bbox[1])
                                x2 = int(bbox[2])
                                y2 = int(bbox[3])
                                label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                                draw_caption(img, (x1, y1, x2, y2), label_name)

                                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                                print(label_name)
                        cv2.imwrite(parser.vis_dir + "/0/" + str(count)+'.jpg',img)
                        #cv2.imshow('img', img)
                        #cv2.waitKey(0)
                        count += 1
        
        #version without nms
        model.duplicate_version = 1
        count = 0
        for idx, data in enumerate(dataloader_val):

                with torch.no_grad():
                        st = time.time()
                        scores, classification, transformed_anchors, anchor_source = retinanet(data['img'].cuda().float())
                        scores = scores.cpu()
                        classification = classification.cpu()
                        print('Elapsed time: {}'.format(time.time()-st))
                        idxs = np.where(scores>0.5*0.5)
                        img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

                        img[img<0] = 0
                        img[img>255] = 255

                        img = np.transpose(img, (1, 2, 0))

                        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

                        for j in range(idxs[0].shape[0]):
                                bbox = transformed_anchors[idxs[0][j], :]
                                x1 = int(bbox[0])
                                y1 = int(bbox[1])
                                x2 = int(bbox[2])
                                y2 = int(bbox[3])
                                label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                                draw_caption(img, (x1, y1, x2, y2), label_name)

                                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                                print(label_name)
                        cv2.imwrite(parser.vis_dir + "/1/" + str(count)+'.jpg',img)
                        #cv2.imshow('img', img)
                        #cv2.waitKey(0)
                        count += 1
        # groundtruth
        count = 0
        for idx, data in enumerate(dataloader_val):

                with torch.no_grad():
                        st = time.time()
                        scores, classification, transformed_anchors, anchor_source = retinanet(data['img'].cuda().float())
                        scores = scores.cpu()
                        classification = classification.cpu()
                        print('Elapsed time: {}'.format(time.time()-st))
                        this_data = data['annot'][0]
                        img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

                        img[img<0] = 0
                        img[img>255] = 255

                        img = np.transpose(img, (1, 2, 0))

                        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

                        for j in range(this_data.shape[0]):
                                bbox = this_data[j][:4]
                                x1 = int(bbox[0])
                                y1 = int(bbox[1])
                                x2 = int(bbox[2])
                                y2 = int(bbox[3])
                                label_name = dataset_val.labels[int(this_data[j][4])]
                                draw_caption(img, (x1, y1, x2, y2), label_name)

                                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                                print(label_name)
                        cv2.imwrite(parser.vis_dir + "/ground_truth/" + str(count)+'.jpg',img)
                        #cv2.imshow('img', img)
                        #cv2.waitKey(0)
                        count += 1


if __name__ == '__main__':
 main()

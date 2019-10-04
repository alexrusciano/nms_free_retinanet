import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
import losses
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import coco_eval
import csv_eval
#import tensorboardX
import sys


import torch.multiprocessing as mp
from losses import DuplicateLoss1, DuplicateLoss2

print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):

    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--version', help ='new thing', type=int, default = 1)
    parser.add_argument('--dup_version', help='nms, alternative1, 2, ...', type=int, default=0)
    parser.add_argument('--batch_update',help='accumulate this many samples before parameter update', type=int, default=16)
    parser.add_argument('--batch_size',help='samples to load at a time', type=int,default = 2)
    parser.add_argument('--testing', help = "set to 1 to test", type = int, default = 0)
    parser.add_argument('--weights', help = 'load these weights', type = str)
    parser = parser.parse_args(args)

#    if not os.path.exists("log_dir"):
#        os.mkdir("log_dir")
#    writer = tensorboardX.SummaryWriter(logdir = 'log_dir', comment = 'training')

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='reduced', transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='reduced', transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

            if parser.csv_train is None:
                    raise ValueError('Must provide --csv_train when training on COCO,')

            if parser.csv_classes is None:
                    raise ValueError('Must provide --csv_classes when training on COCO,')


            dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

            if parser.csv_val is None:
                    dataset_val = None
                    print('No validation annotations provided.')
            else:
                    dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
            raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=4, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
            sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
            dataloader_val = DataLoader(dataset_val, num_workers=2, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50 and parser.version == 0:
        print("old\n")
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50 and parser.version == 1:
        print("new\n")
        retinanet = model.resnet50_op(num_classes=dataset_train.num_classes(), pretrained=True, duplicate_version = parser.dup_version)
        if parser.weights:
            retinanet.load_state_dict(torch.load(parser.weights), strict=False)
        #retinanet.load_state_dict(torch.load("checkpoints/voc_2012/csv_retinanet_20.pt"))
        #retinanet=torch.load("checkpoints/voc_2012/csv_retinanet_20.pt")
        #retinanet = torch.load("checkpoints/coco/coco_retinanet_466.pt")

    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet = torch.nn.DataParallel(retinanet).cuda()

    retinanet.training = True
    #optimizer = optim.SGD(retinanet.parameters(),lr=1e-1,momentum=.9,weight_decay=1e-4)
    optimizer = optim.SGD(retinanet.module.duplicateModel.parameters(),lr=1e-1,momentum=.9,weight_decay=1e-4)
    #optimizer=optim.Adam(retinanet.module.duplicateModel.parameters(), 1e-5)
    #optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    #optimizer = optim.SGD(retinanet.module.duplicateModel.parameters(),lr=1e-2,momentum=.9,weight_decay=1e-4)

    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[300,600],gamma=.1)
    dup_loss_hist = collections.deque(maxlen=500)
    class_loss_hist = collections.deque(maxlen=500)
    regr_loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    if parser.testing == 1:
        if parser.dataset == 'csv':
            mAP = csv_eval.evaluate(dataset_val, retinanet)
        if parser.dataset == 'coco':
            coco_eval.evaluate_coco(dataset_val, retinanet)
        exit(0)

    for epoch_num in range(parser.epochs):
        last_time = time.time()
        retinanet.train()
        retinanet.module.freeze_bn()
        epoch_loss = []

        processed=0

        for iter_num, data in enumerate(dataloader_train):
            very_start = time.time()
            duplicate_loss=0
            loss = 0
            class_labels_out=0
            dup_scores_out=0
            iou_info=0
            if parser.version == 1:
                class_loss, regr_loss, iou_info, class_labels_out, dup_scores_out , anchor_out = retinanet([data['img'].cuda().float(),data['annot'].cuda()])
                regr_loss = regr_loss.mean()
                class_loss = class_loss.mean()
                #start_time = time.time()

                duplicate_loss = DuplicateLoss2(class_labels_out, dup_scores_out, iou_info, anchor_out, data['annot'])
                
                end_time = time.time()
                #print("CPU PART", end_time-start_time)

                loss = class_loss + regr_loss + duplicate_loss

            if parser.version == 0:
                class_loss, regr_loss, = retinanet([data['img'].cuda().float(),data['annot'].cuda()])
                regr_loss = regr_loss.sum()
                class_loss = class_loss.sum()
                loss = class_loss + regr_loss

            loss.backward()
            processed += parser.batch_size
            
            if processed >= parser.batch_update:
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()
                optimizer.zero_grad()
                processed=0


            #very_end = time.time()
            #print("TOTAL", very_end-very_start)
            dup_loss_hist.append(float(duplicate_loss))
            class_loss_hist.append(float(class_loss))
            regr_loss_hist.append(float(regr_loss))

            epoch_loss.append(float(loss))

            if iter_num % 50 == 0:
                print('Epoch: {} | Iteration: {} | sec: {:1.5f} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Dup loss: {:1.5f}'.format(epoch_num, iter_num,  float(time.time()-last_time), np.mean(class_loss_hist), np.mean(regr_loss_hist), np.mean(dup_loss_hist)))

                sys.stdout.flush()
                last_time = time.time()

            del duplicate_loss
            del class_loss, regr_loss
            del loss
            del class_labels_out
            del dup_scores_out

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            if epoch_num % 100 == 0 and epoch_num > -1:
                torch.save(retinanet.module.state_dict(), 'checkpoints/{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))
                coco_eval.evaluate_coco(dataset_val, retinanet)
        elif parser.dataset == 'csv' and parser.csv_val is not None:

            if epoch_num % 5 == 0 or epoch_num == parser.epochs:
                torch.save(retinanet.module.state_dict(), 'checkpoints/{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))
                print('Evaluating dataset')
                mAP = csv_eval.evaluate(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))

    retinanet.eval()


if __name__ == '__main__':
    main()
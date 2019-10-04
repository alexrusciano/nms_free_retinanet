# NMS free variant of Retinanet

Non-maximial suppression is used by most object detectors.  It is quite simple and effective, so it is popular.  However, it does make mistakes and it is a hardcoded heuristic, which is not theoretically satisfying.  It is perhaps particularly unsatisfying for one-pass detectors, as NMS is something of a second pass!  Thus some research has gone into replacing this component of the object detection pipeline.

## Overview
This repository's code is based on 
<https://github.com/yhenon/pytorch-retinanet> ,
which is a popular pytorch implementation of retinanet.

The change to the model is minimal.  Retinanet typically has two "network heads", one for regression of bounding boxes, and one for classification of these bounding boxes.  This code simply adds on a third, outputing a probability an object is a duplicate. The novel addition is in the design of a loss function to train this additional probability.  

For this new loss, roughly speaking, the current code goes through the mAP calculation to assign the model's output bounding boxes labels of 'true object' 'duplicate object' or 'no object'.  Duplicates far outnumber true objects, so we use something resembling hard-negative mining to balance the loss function.  There is flexibility in designing the loss, but this code uses cross-entropy on the latter 2 label categories, and ignores the third class from the loss.

### Prerequisites
Follow the instructions from yhenon's repository, 
<https://github.com/yhenon/pytorch-retinanet> .

Download and parse VOC data into a csv format
(TODO: add these scripts)

## Running
The code can be run through coco_trains.sh or by csv_trains.sh

The former trains on coco, and uses pycocotools to find the coco definition of mAP.  The latter can be modified to run on any dataset formatted in the csv format described in yhenon's repository, but we have used it for the voc dataset.  The mAP for this version is the VOC-Pascal mAP definition.

You can train the entire model from scratch, but part of the idea is that the "duplicate head" can be trained independently, and thus can be initialized with a pretrained retinanet model.
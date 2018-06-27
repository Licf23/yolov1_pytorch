# -*- encoding:utf-8 -*-
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
from data import YoloDataloader
from model import vgg16
import torchvision.transforms as transforms
import cv2
import numpy as np
import time

'''
detect target
@author handsome
'''
Threshold = 0.9
VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
) 

os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

def decoder(pred):
    '''
    pred(tensor) 1x7x7x30
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''

    boxes = []
    cls_indexes = []
    probs = []
    cell_size = 1./7
    pred = pred.data.squeeze(0)     # from 1x7x7x30 to 7x7x30
    contain1 = pred[:,:,4].unsqueeze(2)
    contain2 = pred[:,:,9].unsqueeze(2)
    contain = torch.cat((contain1,contain2),2)    #  7x7x2
    mask1 = contain1 > Threshold           # over threshold
    mask2 = (contain == contain.max())    # select the best one
    mask = (mask1 + mask2).gt(0)          # ensure 1+1 = 1
    min_score,min_index = torch.min(mask,2)   # only choose the max one of two boxes 

    for i in range(7):
        for j in range(7):
            for b in range(2):
                index = min_index[i,j]
                mask[i,j,index] = 0
                if mask[i,j,b] == 1:
                    box = pred[i,j,5*b:5*b+4]
                    contain_prob = torch.FloatTensor([pred[i,j,5*b+4]])
                    xy = torch.FloatTensor([j,i]) * cell_size  ## up left coordiance
                    box[:2] = box[:2] * cell_size + xy
                    box_xy = torch.FloatTensor(box.size())
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]
                    max_prob,cls_index = torch.max(pred[i,j,10:],0)
                    box_xy = torch.clamp(box_xy,min = 0,max = 1)
                    boxes.append(box_xy.view(1,4))
                    cls_indexes.append(cls_index)
                    probs.append(contain_prob)
    boxes = torch.cat(boxes,0)
    probs = torch.cat(probs,0)
    cls_indexes = torch.cat(cls_indexes,0)
    belong = VOC_CLASSES[cls_indexes[0]]
    # print(boxes,probs,cls_indexes,belong)
    keep = nms(boxes,probs) 
    return boxes[keep],cls_indexes[keep],probs[keep]


def nms(bboxes,scores,threshold = 0.5):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,1]
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2 - x1) * (y2 - y1)

    _,order = scores.sort(0,descending = True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min = x1[i])
        yy1 = y1[order[1:]].clamp(min = y1[i])
        xx2 = x2[order[1:]].clamp(max = x2[i])
        yy2 = y2[order[1:]].clamp(max = y2[i])

        w = (xx2 - xx1).clamp(min = 0)
        h = (yy2 - yy1).clamp(min = 0)
        inter = w*h

        ovr = inter/(areas[i] + areas[order[1:]] - inter)
        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)

def predict_gpu(model,image_name,root_path = ""):

    result = []
    image = cv2.imread(root_path+image_name)
    h,w,_ = image.shape
    img = cv2.resize(image,(448,448))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0)

    img = Variable(img[None,:,:,:],volatile = True)
    img = img.cuda()

    pred = model(img)    # 1x7x7x30
    pred = pred.view(1,7,7,30)
    pred = pred.cpu()
    boxes,cls_indexes,probs = decoder(pred)

    for i,box in enumerate(boxes):
        x1 = int(box[0]*w)
        x2 = int(box[2]*w)
        y1 = int(box[1]*h)
        y2 = int(box[3]*h)

        cls_index = cls_indexes[i]
        cls_index = int(cls_index)    # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1,y1),(x2,y2),VOC_CLASSES[cls_index],image_name,prob])
    return result

if __name__ == "__main__":
    ## trained model dir
    model_path = "../yolov1_darknet_pytorch_100.pkl"
    model = torch.load(model_path)
    model.eval()
    model.cuda()
    ## valization dir
    txt_dirs = "../results/train_val.txt"
    ## valization image root
    img_dirs = "/home/xzh/codedata/VOCdevkit/VOC2012/JPEGImages/"
    ## save results dir
    save_dirs = "./train_val_results/"
    with open(txt_dirs,"r") as fn:
        for line in fn.readlines():
                splited = line.strip().split()
                img_path = splited[0]
                img_files_dirs = os.path.join(img_dirs,img_path)
                starttime = time.time()
                image = cv2.imread(img_files_dirs)
                result = predict_gpu(model,img_files_dirs)
                print(time.time()-starttime)  ## test time

                for left_up,right_bottom,class_name,_,prob in result:
                    cv2.rectangle(image,left_up,right_bottom,(0,255,0),2)
                    cv2.putText(image,class_name,left_up,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)

                save_path = os.path.join(save_dirs,img_path)
                cv2.imwrite(save_path,image)
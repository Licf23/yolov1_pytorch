# -*- coding:utf-8 -*-
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import utils
from torchvision import models
from torch.autograd import Variable
from utils import Visualizer 
from data import YoloDataloader
from utils import Lossv1
from darknet_pytorch_full import init_model

'''
yolov1_demo train
@author handsome
'''

## choose GPU  
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

use_gpu = torch.cuda.is_available()
train_img_root = "/home/xzh/codedata/VOCdevkit/VOC2012/JPEGImages/"  ## train image root
train_label_path = "/home/handsome/pycode/yolov1_pytorch_demo/voc_2012_train.txt" ## train label dir

val_img_root = "/home/xzh/codedata/VOCdevkit/VOC2012/JPEGImages/"    ## valization image root 
val_label_path = "/home/handsome/pycode/yolov1_pytorch_demo/voc_2012_val.txt" ## valization label dir

Ceils = 7
Box = 2
coor_l = 5
noor_l = 0.5
learning_rate = 1e-5
momentum = 0.9
decay = 5e-4
epoch_num = 101
batch_size = 8
mean_rgb = [0.485,0.456,0.406]
std_rgb = [0.229,0.224,0.225]
weight_dirs = "/home/handsome/pycode/yolov1_pytorch_demo/yolov1.weights"  ## yolov1 weights file dir

trans = []  ## Image transfrom 

print("Loading pre_trained model ...")

## load initialize model
net = init_model(weight_dirs)
## copy conv weights from darknet to pytorch net 
for i,p in enumerate(net.parameters()):
    if i < 72:
        p.requires_grad = False

if use_gpu:
    net = net.cuda()

print(net)

train_data = YoloDataloader(img_root = train_img_root,label_path = train_label_path,train = True,transforms = trans)
train_dataloader = torch.utils.data.DataLoader(train_data,batch_size= batch_size,shuffle=True,num_workers=4)
val_data = YoloDataloader(img_root = val_img_root,label_path = val_label_path,train = False,transforms = trans)
val_dataloader = torch.utils.data.DataLoader(val_data,batch_size= batch_size,shuffle=False,num_workers=4)

## define yolov1 loss 
criterion = Lossv1(Ceils,Box,coor_l,noor_l)
## only optimize full connect weights
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,net.parameters()),
            lr = learning_rate,momentum=momentum,weight_decay=decay)
vis = utils.Visualizer(env="handsome")   ## visdom 

def train(epoch):
    net.train()
    total_loss = 0.
    for index,(imgs,labels) in enumerate(train_dataloader):
        if use_gpu:
            imgs = imgs.cuda()
            labels = labels.cuda()
        imgs = Variable(imgs)
        labels = Variable(labels)

        optimizer.zero_grad()
        out = net(imgs)
        out = out.view(batch_size,Ceils,Ceils,30)
        loss = criterion(out,labels)    
        loss.backward()
        optimizer.step()
        print("Loss",loss.data[0])
        total_loss += loss.data[0]
        if (index+1)%75 == 0:
            print("Train epoch [%d/%d],iter [%d/%d],lf %.6f,aver_loss %.6f"
                %(epoch,epoch_num,index,len(train_dataloader),learning_rate,total_loss/(index+1)))
            vis.plot_train_val(loss_train = total_loss/(index+1))
    total_loss /= len(train_dataloader)
    print("Train epoch [%d/%d] average_loss %.6f"
            %(epoch,epoch_num,total_loss))

def val(epoch):
    net.eval()
    total_loss = 0.
    for index,(imgs,labels) in enumerate(val_dataloader):
        if use_gpu:
            imgs = imgs.cuda()
            labels = labels.cuda()
        imgs = Variable(imgs,volatile = True)
        labels = Variable(labels,volatile = True)
        out = net(imgs)
        out = out.view(batch_size,Ceils,Ceils,30)
        loss = criterion(out,labels)
        print("Loss",loss.data[0])
        total_loss += loss.data[0]
        if (index+1)%20 == 0:
            print("Val epoch [%d/%d],iter [%d/%d],lf %.6f,aver_loss %.6f"
                %(epoch,epoch_num,index,len(val_dataloader),learning_rate,total_loss/(index+1)))
    total_loss /= len(val_dataloader)
    vis.plot_train_val(loss_val = total_loss)
    print("Val epoch [%d/%d] average_loss %.6f"
            %(epoch,epoch_num,total_loss))


if __name__ == "__main__":
    for epoch in range(epoch_num):
        train(epoch)
        val(epoch)
        model_dir = "yolov1_darknet_pytorch" + str(epoch) + ".pkl"            
        
        if epoch == 10:
            learning_rate *= 0.1
            optimizer.param_groups[0]["lr"] = learning_rate

        if epoch == 50:
            learning_rate *= 0.1
            optimizer.param_groups[0]["lr"] = learning_rate
            torch.save(net,model_dir)
            print(model_dir)

        if epoch == 100:
            torch.save(net,model_dir)
            print(model_dir)
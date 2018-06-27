## -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from torch.autograd import Variable

'''
build pytorch_yolov1_net
@author handsome
'''

## load weights from darknet to pytorch
def load_conv_bn(buf,start,conv_model,bn_model):
    num_conv_w = conv_model.weight.numel()     ## number of conv weights' parameters
    num_bn = bn_model.bias.numel()             ## number of bn weight/bias/running_mean/running_var 'parameters

    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_bn]))
    start += num_bn
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_bn]))
    start += num_bn
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start+num_bn]))
    start += num_bn
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start+num_bn]))
    start += num_bn

    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_conv_w]))
    start += num_conv_w
    
    return start

def load_conv(buf,start,conv_model):
    num_conv_w = conv_model.weight.numel()     ## number of conv weights' parameters
    num_conv_b = conv_model.bias.numel()       ## number of conv bias' parameters
    print(num_conv_b)

    conv_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_conv_b]))   
    start = start + num_conv_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_conv_w]))
    start = start + num_conv_w
    return start

def load_fc(buf,start,fc_model):
    num_w = fc_model.weight.numel()
    num_b = fc_model.bias.numel()
    print(num_w)
    print(num_b)
    fc_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]))     
    start = start + num_b
    fc_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w]))  
    start = start + num_w 
    return start

## build pytorch_yolov1_net
class YoloNet(nn.Module):
    def __init__(self):
        super(YoloNet,self).__init__()

        self.conv1 = nn.Conv2d(3,64,7,stride=2,padding=3,bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpooling = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(64,192,3,stride=1,padding=1,bias = False)
        self.bn2 = nn.BatchNorm2d(192)

        self.conv3 = nn.Conv2d(192,128,1,stride=1,padding=0,bias = False)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128,256,3,stride=1,padding=1,bias = False)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256,256,1,stride=1,padding=0,bias = False)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256,512,3,stride=1,padding=1,bias = False)
        self.bn6 = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(512,256,1,stride=1,padding=0,bias = False)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256,512,3,stride=1,padding=1,bias = False)
        self.bn8 = nn.BatchNorm2d(512)

        self.conv9 = nn.Conv2d(512,256,1,stride=1,padding=0,bias = False)
        self.bn9 = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(256,512,3,stride=1,padding=1,bias = False)
        self.bn10 = nn.BatchNorm2d(512)

        self.conv11 = nn.Conv2d(512,256,1,stride=1,padding=0,bias = False)
        self.bn11 = nn.BatchNorm2d(256)
        self.conv12 = nn.Conv2d(256,512,3,stride=1,padding=1,bias = False)
        self.bn12 = nn.BatchNorm2d(512)

        self.conv13 = nn.Conv2d(512,256,1,stride=1,padding=0,bias = False)
        self.bn13 = nn.BatchNorm2d(256)
        self.conv14 = nn.Conv2d(256,512,3,stride=1,padding=1,bias = False)
        self.bn14 = nn.BatchNorm2d(512)

        self.conv15 = nn.Conv2d(512,512,1,stride=1,padding=0,bias = False)
        self.bn15 = nn.BatchNorm2d(512)

        self.conv16 = nn.Conv2d(512,1024,3,stride=1,padding=1,bias = False)
        self.bn16 = nn.BatchNorm2d(1024)

        self.conv17 = nn.Conv2d(1024,512,1,stride=1,padding=0,bias = False)
        self.bn17 = nn.BatchNorm2d(512)
        self.conv18 = nn.Conv2d(512,1024,3,stride=1,padding=1,bias = False)
        self.bn18 = nn.BatchNorm2d(1024)

        self.conv19 = nn.Conv2d(1024,512,1,stride=1,padding=0,bias = False)
        self.bn19 = nn.BatchNorm2d(512)
        self.conv20 = nn.Conv2d(512,1024,3,stride=1,padding=1,bias = False)
        self.bn20 = nn.BatchNorm2d(1024)

        self.conv21 = nn.Conv2d(1024,1024,3,stride=1,padding=1,bias = False)
        self.bn21 = nn.BatchNorm2d(1024)
        self.conv22 = nn.Conv2d(1024,1024,3,stride=2,padding=1,bias = False)
        self.bn22 = nn.BatchNorm2d(1024)

        self.conv23 = nn.Conv2d(1024,1024,3,stride=1,padding=1,bias = False)
        self.bn23 = nn.BatchNorm2d(1024)
        self.conv24 = nn.Conv2d(1024,1024,3,stride=1,padding=1,bias = False)
        self.bn24 = nn.BatchNorm2d(1024)

        self.fc1 = nn.Linear(7*7*1024,4096)
        self.fc2 = nn.Linear(4096,7*7*30)

    def load_weight(self,weight_file):
        with open(weight_file,"rb") as fp:
            header = np.fromfile(fp,dtype = np.int32,count = 4)
            header = torch.from_numpy(header)
            buf = np.fromfile(fp,dtype = np.float32)
            print(buf.shape)
            start = 0
            start = load_conv_bn(buf,start,self.conv1,self.bn1)
            start = load_conv_bn(buf,start,self.conv2,self.bn2)
            start = load_conv_bn(buf,start,self.conv3,self.bn3)
            start = load_conv_bn(buf,start,self.conv4,self.bn4)
            start = load_conv_bn(buf,start,self.conv5,self.bn5)
            start = load_conv_bn(buf,start,self.conv6,self.bn6)
            start = load_conv_bn(buf,start,self.conv7,self.bn7)
            start = load_conv_bn(buf,start,self.conv8,self.bn8)
            start = load_conv_bn(buf,start,self.conv9,self.bn9)
            start = load_conv_bn(buf,start,self.conv10,self.bn10)
            start = load_conv_bn(buf,start,self.conv11,self.bn11)
            start = load_conv_bn(buf,start,self.conv12,self.bn12)
            start = load_conv_bn(buf,start,self.conv13,self.bn13)
            start = load_conv_bn(buf,start,self.conv14,self.bn14)
            start = load_conv_bn(buf,start,self.conv15,self.bn15)
            start = load_conv_bn(buf,start,self.conv16,self.bn16)
            start = load_conv_bn(buf,start,self.conv17,self.bn17)
            start = load_conv_bn(buf,start,self.conv18,self.bn18)
            start = load_conv_bn(buf,start,self.conv19,self.bn19)
            start = load_conv_bn(buf,start,self.conv20,self.bn20)
            start = load_conv_bn(buf,start,self.conv21,self.bn21)
            start = load_conv_bn(buf,start,self.conv22,self.bn22)
            start = load_conv_bn(buf,start,self.conv23,self.bn23)
            start = load_conv_bn(buf,start,self.conv24,self.bn24)

    def forward(self,x):
        x = F.leaky_relu(self.bn1(self.conv1(x)),0.1,inplace = True)
        x = self.maxpooling(x)

        x = F.leaky_relu(self.bn2(self.conv2(x)),0.1,inplace = True)
        x = self.maxpooling(x)

        x = F.leaky_relu(self.bn3(self.conv3(x)),0.1,inplace = True)
        x = F.leaky_relu(self.bn4(self.conv4(x)),0.1,inplace = True)
        x = F.leaky_relu(self.bn5(self.conv5(x)),0.1,inplace = True)
        x = F.leaky_relu(self.bn6(self.conv6(x)),0.1,inplace = True)
        x = self.maxpooling(x)

        x = F.leaky_relu(self.bn7(self.conv7(x)),0.1,inplace = True)
        x = F.leaky_relu(self.bn8(self.conv8(x)),0.1,inplace = True)
        x = F.leaky_relu(self.bn9(self.conv9(x)),0.1,inplace = True)
        x = F.leaky_relu(self.bn10(self.conv10(x)),0.1,inplace = True)
        x = F.leaky_relu(self.bn11(self.conv11(x)),0.1,inplace = True)
        x = F.leaky_relu(self.bn12(self.conv12(x)),0.1,inplace = True)
        x = F.leaky_relu(self.bn13(self.conv13(x)),0.1,inplace = True)
        x = F.leaky_relu(self.bn14(self.conv14(x)),0.1,inplace = True)
        x = F.leaky_relu(self.bn15(self.conv15(x)),0.1,inplace = True)
        x = F.leaky_relu(self.bn16(self.conv16(x)),0.1,inplace = True)
        x = self.maxpooling(x)

        x = F.leaky_relu(self.bn17(self.conv17(x)),0.1,inplace = True)
        x = F.leaky_relu(self.bn18(self.conv18(x)),0.1,inplace = True)
        x = F.leaky_relu(self.bn19(self.conv19(x)),0.1,inplace = True)
        x = F.leaky_relu(self.bn20(self.conv20(x)),0.1,inplace = True)
        x = F.leaky_relu(self.bn21(self.conv21(x)),0.1,inplace = True)
        x = F.leaky_relu(self.bn22(self.conv22(x)),0.1,inplace = True)
        x = F.leaky_relu(self.bn23(self.conv23(x)),0.1,inplace = True)
        x = F.leaky_relu(self.bn24(self.conv24(x)),0.1,inplace = True)
        
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = F.dropout(x)
        x = self.fc2(x)

        return x

# weight_file = "/home/li/pytest/yolov1_demo/yolov1.weights"

# def main():
#     model = YoloNet()
#     model.load_weight(weight_file)
#     # img = torch.Tensor(1,3,448,448)
#     # img = Variable(img)
#     print(model.conv1.weight.data)
#     print(model.bn1.weight.data)
#     print(model.bn1.bias.data)
#     print(model.bn1.running_mean)
#     print(model.bn1.running_var)


# if  __name__ == "__main__":
#     main()
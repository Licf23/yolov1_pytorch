## -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from load_weights_full import YoloNet

'''
load initialize model
@author handsome
'''
def init_model(weight_dirs):
    darknet_init_model = YoloNet()
    darknet_init_model.load_weight(weight_dirs)
    
    for m in darknet_init_model.modules():
        if isinstance(m,nn.Linear):
            m.weight.data.normal_(0,0.01)
            m.bias.data.zero_()
    
    return darknet_init_model

# weight_dirs = "/home/li/pytest/yolov1_demo/yolov1.weights"
# net = init_model(weight_dirs)
# for i,p in enumerate(net.parameters()):
#     print(i)
    # if i == 71:
    #     print(p)
    # if i == 73:
    #     print(p)

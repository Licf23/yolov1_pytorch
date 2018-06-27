# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable

'''
define own loss
@author handsome
'''

class Lossv1(nn.Module):
    def __init__(self,S,B,coor_l,ncoor_l):
        super(Lossv1,self).__init__()
        self.S = S
        self.B = B
        self.coor_l = coor_l
        self.ncoor_l = ncoor_l

    def forward(self,inputs_tensor,target_tensor):     # [batch_size,7,7,30]
        coo_mask = target_tensor[:,:,:,4] > 0
        noo_mask = target_tensor[:,:,:,4] == 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        coo_mask_pred = inputs_tensor[coo_mask]
        noo_mask_pred = inputs_tensor[noo_mask]  
        coo_pred = coo_mask_pred.view(-1,30)
        noo_pred = noo_mask_pred.view(-1,30)
        
        ## noobj score
        noo_mask_target = target_tensor[noo_mask]
        noo_target = noo_mask_target.view(-1,30)
        noo_target_loc = noo_target[:,:10].contiguous().view(-1,5)
        # print(noo_target_loc)
        noo_target_score = noo_target_loc[:,4]

        noo_pred_loc = noo_pred[:,:10].contiguous().view(-1,5)
        # print(noo_pred_loc)
        noo_pred_score = noo_pred_loc[:,4]
        noobj_score_loss = F.mse_loss(noo_pred_score,noo_target_score,size_average=False)/2.0

        ## coobj score
        coo_mask_target = target_tensor[coo_mask]
        coo_target = coo_mask_target.view(-1,30)
        coo_target_loc = coo_target[:,:10].contiguous().view(-1,5)
        # print(coo_target_loc)
        coo_target_score = coo_target_loc[:,4]
        coo_target_coordiance = coo_target_loc[:,:4]

        coo_pred_loc = coo_pred[:,:10].contiguous().view(-1,5)
        # print(coo_pred_loc)
        coo_pred_score = coo_pred_loc[:,4]
        coo_pred_coordiance = coo_pred_loc[:,:4]

        coobj_score_loss = F.mse_loss(coo_pred_score,coo_target_score,size_average=False)/2.0
        sigmoid_pred = 1/(1 + torch.exp(-coo_pred_coordiance[:,2:]))
        sigmoid_target = 1/(1 + torch.exp(-coo_target_coordiance[:,2:]))
        # sigmoid_pred = coo_pred_coordiance[:,2:]
        # sigmoid_target = coo_target_coordiance[:,2:]
        coobj_coordiance_loss = F.mse_loss(coo_pred_coordiance[:,:2],coo_target_coordiance[:,:2],size_average=False)/2.0 + \
                                F.mse_loss(torch.sqrt(sigmoid_pred),torch.sqrt(sigmoid_target),size_average=False)/2.0
        # coobj_coordiance_loss = F.mse_loss(coo_pred_coordiance[:,:2],coo_target_coordiance[:,:2],size_average=False)/2.0 + \
        #                         F.mse_loss(sigmoid_pred,sigmoid_target,size_average=False)/2.0
    
        ## confidence
        coo_target_confidence = coo_target[:,10:]
        coo_pred_confidence = coo_pred[:,10:]
        # print(coo_target_confidence)
        # print(coo_pred_confidence)
        confidence_loss = F.mse_loss(coo_pred_confidence,coo_target_confidence,size_average=False)/2.0

        loss = self.coor_l * coobj_coordiance_loss + coobj_score_loss + self.ncoor_l * noobj_score_loss + confidence_loss
        return loss

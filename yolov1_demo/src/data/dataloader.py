# -*- coding:utf-8 -*-
import os
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
import random

'''
dataloader dataset including data strengthen
@author handsome 
'''
Image_width = 448
Image_height = 448
Number = 7
CeilSize = Image_width/Number
mean_rgb = [0.485,0.456,0.406]
std_rgb = [0.229,0.224,0.225]

class YoloDataloader(data.Dataset):

    def __init__(self,img_root,label_path,train,transforms):
        super(YoloDataloader,self).__init__()

        print("Data init ...")
        self.img_root = img_root
        self.label_path = label_path
        self.transform = transforms
        self.train = train
        self.imgpath = []
        self.boxes = []
        self.classfiction_label = []

        with open(label_path,"r") as f:
            for line in f.readlines():
                splited = line.strip().split()
                num_faces = int(splited[1])
                img_path = splited[0]
                img_file_dirs = os.path.join(self.img_root,img_path)
                box = []
                c = []
                for i in range(num_faces):
                    x0 = int(splited[5 * i + 2])
                    y0 = int(splited[5 * i + 3])
                    x1 = int(splited[5 * i + 4])
                    y1 = int(splited[5 * i + 5])
                    conf = int(splited[5 * i + 6])
                    box.append([x0,y0,x1,y1])
                    c.append(int(conf))

                self.imgpath.append(img_file_dirs)
                self.boxes.append(torch.Tensor(box))
                self.classfiction_label.append(torch.LongTensor(c))


    def __getitem__(self,idx):
        data_file = self.imgpath[idx]
        img = cv2.imread(data_file)
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)         #img(rgb)
        boxes = self.boxes[idx]
        boxes = boxes * torch.Tensor([1,1,1,1]).expand_as(boxes)
        boxes = boxes.squeeze(1)
        labels = self.classfiction_label[idx]
        
        if self.train:  # data strengthen
            img = self.RandomBrightness(img)
            img = self.RandomSaturation(img)
            img = self.randomBlur(img)
            img,boxes = self.randomShift(img,boxes)
            img,boxes = self.randomCrop(img,boxes)
            img,boxes = self.random_flip(img,boxes)
        # for num in range(len(boxes)):
        #     cv2.rectangle(img,(int(boxes[num,0]),int(boxes[num,1])),(int(boxes[num,2]),int(boxes[num,3])),(255,0,0),3)
        # cv2.imwrite("/home/li/pytest/yolov1_demo/1/only.jpg",img)
        
        h,w,_ = img.shape
        boxes[:,0:3:2] = boxes[:,0:3:2]/w     # normalize boxes [0,1]
        boxes[:,1:4:2] = boxes[:,1:4:2]/h
        img = cv2.resize(img,(Image_height,Image_width))
        img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0)

        target = self.convert(boxes,labels)               # from [[x0,y0,x1,y1]...] to (7,7,30)  
        for trans in self.transform:
            img = trans(img)
        return img,target     ## [0,1]
    
    def convert(self,boxes,labels):
        '''
        boxes [[x0,y0,x1,y1],[x0,y0,x1,y1]...]
        labels [[c],[c]...]
        return 7x7x30
        '''
        target = torch.zeros((Number,Number,30))
        wh = boxes[:,2:] - boxes[:,:2]              # width,height of targets
        centerxy = (boxes[:,:2] + boxes[:,2:])/2    # center xy of targets 
        num_target = len(boxes)                     # numbers of targets

        for i in range(num_target):
            centerxy_sample = centerxy[i]
            array_ij = (centerxy_sample*Number).ceil() - 1  # target belong to (i,j) ceils

            target[int(array_ij[1]),int(array_ij[0]),4] = 1
            target[int(array_ij[1]),int(array_ij[0]),9] = 1
            target[int(array_ij[1]),int(array_ij[0]),int(labels[i]) + 10] = 1
            
            point_xy = (array_ij * CeilSize)/Image_height   # ceil top-left coordinate
            relatively_xy = (centerxy_sample - point_xy) * Number

            target[int(array_ij[1]),int(array_ij[0]),2:4] = wh[i]
            target[int(array_ij[1]),int(array_ij[0]),7:9] = wh[i]
            target[int(array_ij[1]),int(array_ij[0]),:2] = relatively_xy    #[0,1]
            target[int(array_ij[1]),int(array_ij[0]),5:7] = relatively_xy 

        return target

    def __len__(self):
        return len(self.imgpath)

    def RGB2HSV(self,img):  ## convert rgb to hsv
        return cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    
    def HSV2RGB(self,img):  ## convert hsv to rgb
        return cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    
    def RandomBrightness(self,img_rgb):  ## random brightness
        if random.random() < 0.1:
            img_hsv = self.RGB2HSV(img_rgb)
            h,s,v = cv2.split(img_hsv)
            adjust_params = random.choice([0.5,1.5])
            v = adjust_params * v
            v = np.clip(v,0,255).astype(img_hsv.dtype)
            img_hsv = cv2.merge((h,s,v))
            img_rgb = self.HSV2RGB(img_hsv)
        return img_rgb

    def RandomSaturation(self,img_rgb):  ## random saturation 
        if random.random() < 0.1:
            img_hsv = self.RGB2HSV(img_rgb)
            h,s,v = cv2.split(img_hsv)
            adjust_params = random.choice([0.5,1.5])
            s = adjust_params * s
            s = np.clip(s,0,255).astype(img_hsv.dtype)
            img_hsv = cv2.merge((h,s,v))
            img_rgb = self.HSV2RGB(img_hsv)
        return img_rgb
    
    def RandomHue(self,img_rgb):   ## random hue
        if random.random() < 0.1:
            img_hsv = self.RGB2HSV(img_rgb)
            h,s,v = cv2.split(img_hsv)
            adjust_params = random.choice([0.5,1.5])
            h = adjust_params * s
            h = np.clip(s,0,255).astype(img_hsv.dtype)
            img_hsv = cv2.merge((h,s,v))
            img_rgb = self.HSV2RGB(img_hsv)
        return img_rgb
    
    def randomBlur(self,img):  ## random blur
        if random.random() < 0.1:
            img = cv2.blur(img,(5,5))
        return img

    def randomShift(self,img_rgb,boxes):  ## random shift image
        
        center = (boxes[:,2:] + boxes[:,:2])/2
        if random.random() < 0.1:
            height,width,c = img_rgb.shape
            after_shift_image = np.zeros((height,width,c),dtype = img_rgb.dtype)
            after_shift_image[:,:,:] = (123,117,104)
            shift_x = random.uniform(-width*0.2,width*0.2)
            shift_y = random.uniform(-height*0.2,height*0.2)

            if shift_x >= 0 and shift_y >= 0:
                after_shift_image[int(shift_y):,int(shift_x):,:] = img_rgb[:height-int(shift_y),:width-int(shift_x),:]
            elif shift_x >= 0 and shift_y < 0:
                after_shift_image[:height+int(shift_y),int(shift_x):,:] = img_rgb[-int(shift_y):,:width-int(shift_x),:]
            elif shift_x < 0 and shift_y >= 0:
                after_shift_image[int(shift_y):,:width+int(shift_x),:] = img_rgb[:height-int(shift_y),-int(shift_x):,:]
            else:
                after_shift_image[:height+int(shift_y),:width+int(shift_x),:] = img_rgb[-int(shift_y):,-int(shift_x):,:]
        
            h,w,_ = after_shift_image.shape
            shift_xy = torch.FloatTensor([[int(shift_x),int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:,0] >0) & (center[:,0] < width)
            mask2 = (center[:,1] >0) & (center[:,1] < height)
            mask = (mask1 & mask2).view(-1,1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if len(boxes_in) == 0:
                return img_rgb,boxes
            box_shift = torch.FloatTensor([[int(shift_x),int(shift_y),int(shift_x),int(shift_y)]]).expand_as(boxes_in)
            boxes_in = (boxes_in+box_shift)
            boxes_in[:,0:3:2] = torch.clamp(boxes_in[:,0:3:2],min = 0,max = w)
            boxes_in[:,1:4:2] = torch.clamp(boxes_in[:,1:4:2],min = 0,max = h)
            return after_shift_image,boxes_in
        return img_rgb,boxes

    def randomScale(self,img_rgb,boxes):   ## random scale image only height
        if random.random() < 0.1:
            scale = random.uniform(0.8,1.2)
            height,width,c = img_rgb.shape
            img_rgb = cv2.resize(img_rgb,(int(width*scale),height))
            h,w,_ = img_rgb.shape
            scale_tensor = torch.FloatTensor([[scale,1,scale,1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            boxes[:,0:3:2] = torch.clamp(boxes[:,0:3:2],min = 0,max = w)
            boxes[:,1:4:2] = torch.clamp(boxes[:,1:4:2],min = 0,max = h)
            return img_rgb,boxes
        return img_rgb,boxes

    def randomCrop(self,img_rgb,boxes):  ## random crop image
        if random.random() < 0.1:
            center = (boxes[:,2:] + boxes[:,:2]) / 2

            height,width,_ = img_rgb.shape
            h = random.uniform(0.8,1) 
            w = random.uniform(0.8,1) 
            x = random.uniform(0,1 - w) 
            y = random.uniform(0,1 - h) 
            x,y,w,h = int(x*width),int(y*height),int(w*width),int(h*height)

            center = center - torch.FloatTensor([[x,y]]).expand_as(center)
            mask1 = (center[:,0] > 0) & (center[:,0] < height)
            mask2 = (center[:,1] > 0) & (center[:,1] < width)
            mask = (mask1 & mask2).view(-1,1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if(len(boxes_in) == 0):
                return img_rgb,boxes
            boxes_shift = torch.FloatTensor([[x,y,x,y]]).expand_as(boxes_in)
            boxes_in = boxes_in - boxes_shift
            img_croped = img_rgb[y:y+h,x:x+w,:]
            h,w,_ = img_croped.shape
            boxes_in[:,0:3:2] = torch.clamp(boxes_in[:,0:3:2],min = 0,max = w)
            boxes_in[:,1:4:2] = torch.clamp(boxes_in[:,1:4:2],min = 0,max = h)
            return img_croped,boxes_in
        return img_rgb,boxes
    
    def subMean(self,img_rgb,mean):
        mean = np.array(mean,dtype=np.float32)
        img_rgb = img_rgb - mean
        return img_rgb
    
    def random_flip(self,img_rgb,boxes):  ## random flip image
        if random.random() < 0.1:
            im_lr = np.fliplr(img_rgb).copy()
            h,w,_ = img_rgb.shape
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
            return im_lr,boxes
        return img_rgb,boxes
    
    def random_bright(self,img_rgb,delta = 16):  ## random bright
        alpha = random.random()
        if alpha > 0.9:
            img_rgb = img_rgb * alpha + random.randrange(-delta,delta)
            img_rgb = img_rgb.clip(min = 0,max = 255).astype(np.uints8)
        return img_rgb
            
# def main():
#     file_root = "/home/li/pytest/yolov1_demo/VOC_2012"
#     label_path = "/home/li/pytest/yolov1_demo/vvoc_2012_1.txt"
#     trans = [
#         # transforms.ToTensor(),
#         # transforms.Normalize(mean_rgb,std_rgb),
#         ]
#     train_dataset = YoloDataloader(img_root = file_root,label_path = label_path,train = True,transforms = trans)
#     train_loader = DataLoader(train_dataset,batch_size=4,shuffle=False,num_workers=0)
#     train_iter = iter(train_loader)
#     img,target = next(train_iter)
#     # print(target[0,3,3])
#     # print(target[0,2,2])
#     # print(target[0,5,2])
#     # print(target[0,5,0])
#     return img,target

# if __name__ == "__main__":
#     main() 
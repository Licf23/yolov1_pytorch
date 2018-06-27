Yolov1_demo_pytorch<br>
===================
This is a yolov1 demo by pytorch,I use the base net from the original struction.However,I use the last two layers by full connected layer.The base network filled with convolution layer use the [weights](http://pjreddie.com/media/files/yolov1/yolov1.weights) from author,we train the last two layers by myself.<br>

1.Environment<br>
-------------
* python 2.7<br>
* pytorch 0.3.0<br>
* opencv<br>
* visdom<br>
* tqdm<br>

2.DataSet Prepare<br>
-------------
1.Download voc2012 train dataset.<br>
2.Download voc2007 test dataset.<br>
3.Convert xml annotations to .txt file,like [voc2012.txt](https://github.com/Licf23/yolov1_pytorch/blob/master/yolov1_demo/dataset/voc2012.txt).<br>

3.Train Model<br>
-------------
Run python train.py<br>
You must change the dataset path and model saved dirs.For more details please read train.py<br>
Actually,to see the loss during train,you'd better install [visdom](https://blog.csdn.net/u012436149/article/details/69389610).<br>

4.Test Model<br>
-------------
Run python pred.py<br>
You also need to change the path,including image path and model dirs.For more details please read train.py<br>

5.Results<br>
-------------
First,I draw the training loss and validate loss as follows.<br>
![](https://github.com/Licf23/yolov1_pytorch/blob/master/yolov1_demo/results/loss.png)<br>
Then,Show you the predict result in TrainSet.For more results you could visit [train_val_results](https://github.com/Licf23/yolov1_demo/raw/master/results/train_val_results).<br>
![](https://github.com/Licf23/yolov1_pytorch/blob/master/yolov1_demo/results/train_val_results/2009_004681.jpg)<br>
And in ValidationSet.Also visit [test_val_results](https://github.com/Licf23/yolov1_pytorch/tree/master/yolov1_demo/results/test_val_results).<br>
![](https://github.com/Licf23/yolov1_pytorch/blob/master/yolov1_demo/results/test_val_results/2011_004581.jpg)<br>

6.Discussion<br>
--------------
As you see the results are not satisfied,especially the pictures including multitargets.<br>
In loss.py we don't use the sqrt process as author raised,actually it's not very useful in my experiments.<br>
Thank to [xiong](https://github.com/xiongzihua/pytorch-YOLO-v1).<br>
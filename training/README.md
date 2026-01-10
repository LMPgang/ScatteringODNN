## 1. 文件夹简介
diffuser_data\diff_256  （存放散射介质数据集的位置）
mnist_data\mnist_256  （存放物体数据集的位置）
results  (存放结果。包括图片、衍射网络、备份代码、训练参数等)
errfiles  （存放中间输出参数、错误）
outfiles  （存放中间输出参数）

## 2. 程序简介
* dataset.py:  MNISTDataset用于加载物体图像数据
* diffuser.py: DiffuserProvider用于加载仿真散射介质
* DynaDiffuser_2layer_v2_SLM.py: 代码运行主程序（里面包括衍射网络参数、结构的设置）
* run_SLM.py: 设置训练参数（包括bs，lr，epoch等）
* run_SLM.slurm 提交程序

### 3.程序运行逻辑

第一步、在DynaDiffuser_2layer_v2_SLM.py中设置好基本的网络参数
第二步、在run_SLM.py 设置好基本的训练参数
第三步、用run_SLM.slurm提交程序

备注：如果想演示物体旋转和缩放，请将 `DynaDiffuser_2layer_v2_SLM.py` 中如下代码取消注释

演示旋转
```
# object rotate 90 degree
##----rotate----##
# train_object_image = train_object_image.flip(1).flip(2).contiguous()
# train_object_image = train_object_image.permute(0, 2, 1).flip(2).contiguous()  
############

##----rotate----##
# test_object_image = test_object_image.flip(1).flip(2).contiguous()
# test_object_image = test_object_image.permute(0, 2, 1).flip(2).contiguous()
```

演示缩放
```
# Object scaling
##--scale--##
# train_object_image = train_object_image.unsqueeze(1)  
# train_object_image_interp = F.interpolate(train_object_image, size=(128, 128), mode='bilinear', align_corners=False) 
# train_object_image_padded = F.pad(train_object_image_interp, (64, 64, 64, 64), mode='constant', value=0) 
# train_object_image = train_object_image_padded.squeeze(1) # 
##########

##----scale----##
# test_object_image = test_object_image.unsqueeze(1)  
# test_object_image_interp = F.interpolate(test_object_image, size=(128, 128), mode='bilinear', align_corners=False) 
# test_object_image_padded = F.pad(test_object_image_interp, (64, 64, 64, 64), mode='constant', value=0) 
# test_object_image = test_object_image_padded.squeeze(1) # 
```
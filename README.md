## 代码适用步骤：
## 常规模型训练测试
1. conda  activate  mmpose  #切换环境为mmpose
2. cd castpose
3. python   train.py #用于训练模型
4. python test.py --checkpoint work_dirs/epoch_20.pth #模型为训练好模型的路径
5. python inference.py --checkpoint work_dirs/epoch_20.pth #模型为训练好模型的路径 #用于推理测试集的结果，并保存在results_inference
6. python  deploy.py --checkpoint work_dirs/epoch_20.pth #模型为训练好模型的路径#转化模型为onnx，结果存在results中
## tools
在tools里有相关的辅助的脚本代码，可供使用
## tricks
此文件夹放了自实现的模块，可供选择替换

## 蒸馏模型
在config/distiller里是自己移植的官方的蒸馏的相关配置；
在configs/castpose里有自实现的蒸馏配置

## 训练蒸馏模型命令##
cd mmpose
可以直接运行 train_dis1_rtw(有融合模块)：一阶段蒸馏；
可以直接运行 train_dis2(有融合模块)：二阶段蒸馏；
经验:[一阶段蒸馏+二阶段蒸馏的效果<自训练（监督）+二阶段蒸馏<二阶段蒸馏+二阶段蒸馏]


## 蒸馏模型转正常模型
####first stage 
python dis2nor_pth.py $旧模型权重路径 $新模型权重路径
####second stage distillation
python dis2nor_pth.py $dis_ckpt $new_pose_ckpt --two_dis
对齐neck蒸馏后的模型，可根据自身情况进行一阶段/二阶段的配置
## 项目介绍
此项目基于mmpose，对mmpose做了减重，同时扩展了功能；
#### 1.删去了不必需要的算法及数据集；
#### 2.添加了上半身+双手的关键点检测识别 castpose
#### 3.添加了新的det模型 casthand；
#### 4.加入了自定义实现的模块和功能 
   a. 实现了基于det-pose-track的全流程：在文件夹rtm_det_pose_track里，可能会有报错，持续更新优化

   b. 实现了mmpose的检测轻量化，在文件夹lights里，可以免去繁杂的支撑；

   c. 在configs/castpose的配置里有多种配置，其中添加了可见性预测头(castpose_hand.py)，添加了冻结hook(pose_freezen)等;

   d. 实现自定义融合模块(pose_impove_txy),backbone以及head，可根据自己需求组合使用；

   e. 实现了自定义的姿态估计器(mmpose/models/pose_estimators/top_down_TXYDR.py)
#### 5.数据配置
   针对上半身和双手配置了独有的config及dataset，可根据官方文档进行对应位置查看(config/_base_/datasets;mmpose\datasets\datasets),数据集无法提供，可根据lab2coco文件的脚本进行数据解析和转换
#### 6.在tool中加入了更多便捷脚本

[预训练权重及对应文件下载]下载后放入对应的文件夹中：

链接：https://pan.baidu.com/s/1J3kWnP_8GswjBexYqIxlaw?pwd=4s5c 
提取码：4s5c







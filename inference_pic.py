import cv2
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt


import torch

import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

from mmdet.apis import inference_detector, init_detector

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)
#1.待测图片载入"
img_path = '/home/txy/code/CastPose/pic/Image9.jpg'
#2.构建检测模型
detector = init_detector(
    'configs/castpose/yolox_l_det.py',##config
    'pth/det/yolox_det.pth',###对应的权重文件
    device=device)
#3.构建关键点检测模型
pose_estimator = init_pose_estimator(
    '/home/txy/code/CastPose/configs/castpose/cocktail3.py',
  
    '/home/txy/code/CastPose/pth/castpose/modified_rtw.pth',
  
    
   
    device=device,
    cfg_options={'model': {'test_cfg': {'output_heatmaps': False}}}###如果需要热力图就改为True
 )
#4.预测初始化
init_default_scope(detector.cfg.get('default_scope', 'mmdet'))
#5.获取结果
detect_result = inference_detector(detector, img_path)
# 置信度阈值
CONF_THRES = 0.5

pred_instance = detect_result.pred_instances.cpu().numpy()##也可用gpu进行预测
bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > CONF_THRES)]
bboxes = bboxes[nms(bboxes, 0.3)][:, :4].astype('int')

# 获取每个 bbox 的关键点预测结果
pose_results = inference_topdown(pose_estimator, img_path, bboxes)
data_samples = merge_data_samples(pose_results)

#6.可视化
# ##1opencv
# img_bgr = cv2.imread(img_path)
# #设置
# # 检测框的颜色
# bbox_color = (150,0,0)
# # 检测框的线宽
# bbox_thickness = 20
# # 关键点半径
# kpt_radius = 70
# # 连接线宽
# skeleton_thickness = 30
#
# # 三角板关键点检测数据集-元数据（直接从config配置文件中粘贴）
# dataset_info = {
#     'keypoint_info':{
#         0:{'name':'angle_30','id':0,'color':[255,0,0],'type': '','swap': ''},
#         1:{'name':'angle_60','id':1,'color':[0,255,0],'type': '','swap': ''},
#         2:{'name':'angle_90','id':2,'color':[0,0,255],'type': '','swap': ''}
#     },
#     'skeleton_info': {
#         0: {'link':('angle_30','angle_60'),'id': 0,'color': [100,150,200]},
#         1: {'link':('angle_60','angle_90'),'id': 1,'color': [200,100,150]},
#         2: {'link':('angle_90','angle_30'),'id': 2,'color': [150,120,100]}
#     }
# }
#
# # 关键点类别和关键点ID的映射字典
# label2id = {}
# for each in dataset_info['keypoint_info'].items():
#     label2id[each[1]['name']] = each[0]
#
# for bbox_idx, bbox in enumerate(bboxes):  # 遍历每个检测框
#
#     # 画框
#     img_bgr = cv2.rectangle(img_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, bbox_thickness)
#
#     # 索引为 0 的框，每个关键点的坐标
#     keypoints = data_samples.pred_instances.keypoints[bbox_idx, :, :].astype('int')
#
#     # 画连线
#     for skeleton_id, skeleton in dataset_info['skeleton_info'].items():  # 遍历每一种连接
#         skeleton_color = skeleton['color']
#         srt_kpt_id = label2id[skeleton['link'][0]]  # 起始点的类别 ID
#         srt_kpt_xy = keypoints[srt_kpt_id]  # 起始点的 XY 坐标
#         dst_kpt_id = label2id[skeleton['link'][1]]  # 终止点的类别 ID
#         dst_kpt_xy = keypoints[dst_kpt_id]  # 终止点的 XY 坐标
#         img_bgr = cv2.line(img_bgr, (srt_kpt_xy[0], srt_kpt_xy[1]), (dst_kpt_xy[0], dst_kpt_xy[1]),
#                            color=skeleton_color, thickness=skeleton_thickness)
#
#     # 画关键点
#     for kpt_idx, kpt_xy in enumerate(keypoints):  # 遍历该检测框中的每一个关键点
#         kpt_color = dataset_info['keypoint_info'][kpt_idx]['color']
#         img_bgr = cv2.circle(img_bgr, (kpt_xy[0], kpt_xy[1]), kpt_radius, kpt_color, -1)
# plt.imshow(img_bgr[:,:,::-1])
# plt.show()
#2.官方可视化工具
# 半径
pose_estimator.cfg.visualizer.radius = 2
# 线宽
pose_estimator.cfg.visualizer.line_width = 2
visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
# 元数据
visualizer.set_dataset_meta(pose_estimator.dataset_meta)
img = mmcv.imread(img_path)
img = mmcv.imconvert(img, 'bgr', 'rgb')

img_output = visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=True,
            draw_bbox=True,
            show=False,
            show_kpt_idx=True,
            wait_time=0,
            out_file='pic_test_out/new-rtw.PNG',
            kpt_thr=0.3
)
##输出
plt.figure(figsize=(20, 20))
plt.imshow(img_output)
plt.show()
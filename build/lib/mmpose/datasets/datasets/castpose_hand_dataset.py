# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import Optional

import numpy as np
from mmpose.datasets.datasets.base.base_coco_style_dataset import BaseCocoStyleDataset

from mmpose.registry import DATASETS



@DATASETS.register_module(name="QXCastHandDatasets")
class QXCastHandDatasets(BaseCocoStyleDataset):
    """CocoWholeBody dataset for pose estimation.



    COCO-WholeBody keypoints::

        0-10: 11 body keypoints,
        11-53: 42 hand keypoints,

        In total, we have 53 keypoints for wholebody pose estimation.


    """

    METAINFO: dict = dict(
        from_file='/home/txy/code/CastPose/configs/_base_/datasets/qx_hand_castpose.py')

    # def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
    #     """Parse raw COCO annotation of an instance.

    #     Args:
    #         raw_data_info (dict): Raw data information loaded from
    #             ``ann_file``. It should have following contents:

    #             - ``'raw_ann_info'``: Raw annotation of an instance
    #             - ``'raw_img_info'``: Raw information of the image that
    #                 contains the instance

    #     Returns:
    #         dict: Parsed instance annotation
    #     """

    #     ann = raw_data_info['raw_ann_info']
    #     img = raw_data_info['raw_img_info']

    #     img_path = osp.join(self.data_prefix['img'], img['file_name'])
    #     img_w, img_h = img['width'], img['height']

    #     # get bbox in shape [1, 4], formatted as xywh
    #     x, y, w, h = ann['bbox']
    #     x1 = np.clip(x, 0, img_w - 1)
    #     y1 = np.clip(y, 0, img_h - 1)
    #     x2 = np.clip(x + w, 0, img_w - 1)
    #     y2 = np.clip(y + h, 0, img_h - 1)

    #     bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

    #     # keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
    #     # COCO-Wholebody: consisting of body, foot, face and hand keypoints
    #     _keypoints = np.array(ann['keypoints']).reshape(1, -1, 3)
    #     keypoints = _keypoints[..., :2]
    #     keypoints_visible = np.minimum(1, _keypoints[..., 2] > 0)

    #     num_keypoints = ann['num_keypoints']

    #     data_info = {
    #         'img_id': ann['image_id'],
    #         'img_path': img_path,
    #         'bbox': bbox,
    #         'bbox_score': np.ones(1, dtype=np.float32),
    #         'num_keypoints': num_keypoints,
    #         'keypoints': keypoints,
    #         'keypoints_visible': keypoints_visible,
    #         'iscrowd': ann['iscrowd'],
    #         'segmentation': ann['segmentation'],
    #         'id': ann['id'],
    #         'category_id': ann['category_id'],
    #         # store the raw annotation of the instance
    #         # it is useful for evaluation without providing ann_file
    #         'raw_ann_info': copy.deepcopy(ann),
    #     }

    #     return data_info

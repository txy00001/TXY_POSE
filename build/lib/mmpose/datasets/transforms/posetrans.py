import numpy as np
import cv2
import pycocotools.mask as mask_util
from collections import OrderedDict
import torch
from mmcv import BaseTransform
from mmengine import TRANSFORMS
from torch import nn
###子关节和主关节间的映射，方便转换
PART_TO_KP_IDS = {
    'left_arm': [5, 7, 9],
    'right_arm': [6, 8, 10],
    'face':[0,1,2,3,4],
    'left_hand': [11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
    'right_hand':[32, 33, 34, 35, 36, 37, 38,39,40,41,42,43,44,45,46,47,48,49,50,51,52]
}

PART_TO_SUBPART = {
     'left_arm':['left_shoulder','left_elbow','left_wrist'],
     'right_arm':['right_shoulder','right_elbow','right_wrist'],
     'face':['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'],
     'left_hand':['left_hand_root','left_thumb1','left_thumb2','left_thumb3','left_thumb4',
                  'left_forefinger1','left_forefinger2','left_forefinger3','left_forefinger4','left_middle_finger1',
                   'left_middle_finger2','left_middle_finger3','left_middle_finger4','left_ring_finger1',
                  'left_ring_finger2','left_ring_finger3','left_ring_finger4','left_pinky_finger1',
                    'left_pinky_finger2','left_pinky_finger3','left_pinky_finger4',],
     'right_hand':['right_hand_root','right_thumb1','right_thumb2','right_thumb3','right_thumb4',
                   'right_forefinger1','right_forefinger2','right_forefinger3','right_forefinger4','right_middle_finger1',
                    'right_middle_finger2','right_middle_finger3','right_middle_finger4','right_ring_finger1',
                   'right_ring_finger2','right_ring_finger3','right_ring_finger4','right_pinky_finger1',
                    'right_pinky_finger2','right_pinky_finger3','right_pinky_finger4']
}
SUBPART_TO_PART = {
             ('left_shoulder','left_elbow','left_wrist'): 'left_arm',
             ('right_shoulder','right_elbow','right_wrist'): 'right_arm',
             ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'): 'face',
             ( 'left_hand_root','left_thumb1','left_thumb2','left_thumb3','left_thumb4',
              'left_forefinger1','left_forefinger2','left_forefinger3','left_forefinger4','left_middle_finger1',
              'left_middle_finger2','left_middle_finger3','left_middle_finger4','left_ring_finger1',
              'left_ring_finger2','left_ring_finger3','left_ring_finger4','left_pinky_finger1',
              'left_pinky_finger2','left_pinky_finger3','left_pinky_finger4'): 'left_hand',

             ('right_hand_root','right_thumb1','right_thumb2','right_thumb3','right_thumb4',
                   'right_forefinger1','right_forefinger2','right_forefinger3','right_forefinger4','right_middle_finger1',
                    'right_middle_finger2','right_middle_finger3','right_middle_finger4','right_ring_finger1',
                   'right_ring_finger2','right_ring_finger3','right_ring_finger4','right_pinky_finger1',
                    'right_pinky_finger2','right_pinky_finger3','right_pinky_finger4'): 'right_hand',
}

KP_IDS = list(range(53))



@TRANSFORMS.register_module(name="PoseTrans",force=True)
class PoseTrans(BaseTransform):


    def __init__(self, body_parts=('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
                      'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',

                      'left_hand_root','right_hand_root','left_thumb1','right_thumb1','left_thumb2',
                      'right_thumb2','left_thumb3','right_thumb3','left_thumb4','right_thumb4','left_forefinger1',

                      'right_forefinger1','left_forefinger2','right_forefinger2','left_forefinger3','right_forefinger3',
                      'left_forefinger4','right_forefinger4','left_middle_finger1','right_middle_finger1','left_middle_finger2',

                      'right_middle_finger2','left_middle_finger3','right_middle_finger3','left_middle_finger4','right_middle_finger4',
                      'left_ring_finger1','right_ring_finger1','left_ring_finger2','right_ring_finger2','left_ring_finger3',

                      'right_ring_finger3','left_ring_finger4','right_ring_finger4','left_pinky_finger1','right_pinky_finger1',
                      'left_pinky_finger2','right_pinky_finger2','left_pinky_finger3','right_pinky_finger3','left_pinky_finger4',
                      'right_pinky_finger4'),
                 aug_probabilities=(1,1,1,1,1,1,1,1,1,1,
                                    1,1,1,1,1,1,1,1,1,1,
                                    1,1,1,1,1,1,1,1,1,1,
                                    1,1,1,1,1,1,1,1,1,1,
                                    1,1,1,1,1,1,1,1,1,1,
                                    1,1,1,),
                 rot_factor=35,
                 scale_factor=0.25,
                 T=5,
                 # discriminator = r'C:\Users\ASUS\Desktop\CastPose\pth/discriminator.pth',##加入判别器
                 E=0.7
                 ):
        self.body_parts = body_parts
        self.rot_factor = rot_factor
        self.scale_factor = scale_factor
        self.aug_probabilities = aug_probabilities
        self.T = T
        # self.D = torch.load(discriminator)
        self.E = E

    @staticmethod
    def _is_exist(body_part, results, insta_idx):
        """ Return mask if exist, else return False
            for TopDown method with single instance, insta_idx = 0
        """
        ori_kp_anno = results['ann_detail_info']['keypoints'].reshape(-1, results['ann_info']['num_joints'], 3)

        # the keypoint in the body part should be visible
        for kp_id in PART_TO_KP_IDS[body_part]:
            if ori_kp_anno[insta_idx, kp_id, 2] != 2:
                return False


    @staticmethod
    def _get_3rd_point(a, b):
        """To calculate the affine matrix, three pairs of points are required. This
        function is used to get the 3rd point, given 2D points a & b.

        The 3rd point is defined by rotating vector `a - b` by 90 degrees
        anticlockwise, using b as the rotation center.

        Args:
            a (np.ndarray): point(x,y)
            b (np.ndarray): point(x,y)

        Returns:
            np.ndarray: The 3rd point.
        """
        assert len(a) == 2
        assert len(b) == 2
        direction = a - b
        third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)
        return third_pt

    @staticmethod
    def _rotate_point(pt, angle_rad):
        """Rotate a point by an angle.

        Args:
            pt (list[float]): 2 dimensional point to be rotated
            angle_rad (float): rotation angle by radian

        Returns:
            list[float]: Rotated point.
        """
        assert len(pt) == 2
        sn, cs = np.sin(angle_rad), np.cos(angle_rad)
        new_x = pt[0] * cs - pt[1] * sn
        new_y = pt[0] * sn + pt[1] * cs
        rotated_pt = [new_x, new_y]
        return rotated_pt

    @staticmethod
    def pose_normalize(kpt, bboxes):
        """Normalize the given pose using bounding box.

        Args:
            kpt: (b, kp_num, 2)
            bboxes: (4)

        Returns:
            kpt_new: (b, kp_num, 2)
        """
        xmin, ymin, xmax, ymax = bboxes
        ymin, ymax, xmin, xmax = [int(round(x)) for x in (ymin, ymax, xmin, xmax)]
        h = ymax - ymin
        w = xmax - xmin
        x, y, w, h = xmin, ymin, w, h

        aspect_ratio = 1.0
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        scale = scale * 1.1

        kpt_new = (kpt - center) / scale + np.array([0.5, 0.5])
        kpt_new[..., 1] = 1 - kpt_new[..., 1]  # (b, kp_num, 2)

        return kpt_new

    def _get_affine_transform(self, body_part, results, insta_idx=None):
        """Obtain the transformation matrix given the meta data.

        Args:
            body_part: The current transformed body_part
            results: Dict that contain meta data
            insta_idx: The index of instance in the image (0 for top-down methods)

        Returns:
            trans: Transformation matrix
        """
        if 'joints_3d' in results:  # top down
            s = results['scale']
        else:  # bottom up
            xmin, ymin, xmax, ymax = results['ann_detail_info']['bboxes'][insta_idx]
            ymin, ymax, xmin, xmax = [int(round(x)) for x in (ymin, ymax, xmin, xmax)]
            h = ymax - ymin
            w = xmax - xmin
            scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
            scale = scale * 1.25
            s = scale

        sf = self.scale_factor
        rf = self.rot_factor
        s_factor = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        s = s * s_factor

        r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        center_id = PART_TO_KP_IDS[body_part][0]  # rotation center
        if 'joints_3d' in results:
            center = results['joints_3d'][center_id, :2]
        else:
            center = results['joints'][0][insta_idx, center_id, :2]

        # pixel_std is 200.
        scale_tmp = s * 200.0
        src_w = scale_tmp[0]
        rot_rad = np.pi * r / 180
        src_dir = self._rotate_point([0., src_w * -0.5], rot_rad)
        dst_dir = np.array([0., src_w * -0.5])

        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center
        src[1, :] = center + src_dir
        src[2, :] = self._get_3rd_point(src[0, :], src[1, :])
        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = center
        dst[1, :] = center + dst_dir
        dst[2, :] = self._get_3rd_point(dst[0, :], dst[1, :])

        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return trans

    @staticmethod
    def _affine_transform(pt, trans_mat):
        """Apply an affine transformation to the points.

        Args:
            pt (np.ndarray): a 2 dimensional point to be transformed
            trans_mat (np.ndarray): 2x3 matrix of an affine transform

        Returns:
            new_pt (np.ndarray): Transformed points.
        """
        assert len(pt) == 2
        new_pt = np.array(trans_mat) @ np.array([pt[0], pt[1], 1.])
        return new_pt

    def _transform_img_and_anno_bottomup(self, affine_instance_mask, results):
        """transform the image and annotations for bottom-up methods"""
        joints = results['joints']
        part_mask_sum = np.zeros(results['img'].shape[0:2])

        # ********************** Erase original limbs ***********************
        for insta_idx in affine_instance_mask:
            for body_part in affine_instance_mask[insta_idx]:
                part_mask = affine_instance_mask[insta_idx][body_part]
                xmin, ymin, xmax, ymax = results['ann_detail_info']['bboxes'][insta_idx]
                ymin, ymax, xmin, xmax = [int(round(x)) for x in (ymin, ymax, xmin, xmax)]
                part_mask_sum[ymin:ymax, xmin:xmax] += part_mask
        ret_img = cv2.inpaint(results['img'], np.uint8(part_mask_sum), 5, cv2.INPAINT_NS)

        # ********************** Candidate Pose Pool ***********************
        if self.T > 1:
            trans_dict = OrderedDict()
            for insta_idx in affine_instance_mask:
                trans_list = []
                affined_joints = []
                for idx in range(self.T):
                    cur_e = 0
                    while cur_e < self.E:
                        cur_trans_dict = OrderedDict()
                        cur_joints = np.copy(joints[0][insta_idx, :, 0:2])  # (17, 2)
                        for body_part in affine_instance_mask[insta_idx]:
                            cur_trans_dict[(insta_idx, body_part)] = self._get_affine_transform(body_part, results,
                                                                                                insta_idx=insta_idx)
                            for j in PART_TO_KP_IDS[body_part]:
                                cur_joints[j] = self._affine_transform(cur_joints[j],
                                                                       cur_trans_dict[(insta_idx, body_part)])
                        cur_e = self.D(results['img'], cur_trans_dict, cur_joints)
                    affined_joints.append(cur_joints)  # (17, 2)
                    trans_list.append(cur_trans_dict)

                # predict the probabilities of belonging to each components, then calculate the weighted sum of probabilities
                affined_joints = np.stack(affined_joints)  # (T, 17, 2)
                predict_weights_sum = np.zeros(self.T)
                normal_affined_joints = self.pose_normalize(affined_joints[:, KP_IDS],
                                                            bboxes=results['ann_detail_info']['bboxes'][insta_idx])
                cur_kps = normal_affined_joints.reshape(-1, len(KP_IDS) * 2)
                predict_proba = results['gmms'].predict_proba(cur_kps)  # (b, n_components)
                predict_weights_sum += (results['gmms'].weights_ * predict_proba).sum(1)

                min_idx = np.argmin(predict_weights_sum)
                trans_dict.update(trans_list[min_idx])
        else:
            trans_dict = OrderedDict()
            for insta_idx in affine_instance_mask:
                for body_part in affine_instance_mask[insta_idx]:
                    trans_dict[(insta_idx, body_part)] = self._get_affine_transform(body_part, results,
                                                                                    insta_idx=insta_idx)

        # ******************** Start Transformation *******************
        for insta_idx in affine_instance_mask:
            for body_part in list(affine_instance_mask[insta_idx].keys()):
                part_mask = affine_instance_mask[insta_idx][body_part]
                if part_mask is None:
                    continue
                xmin, ymin, xmax, ymax = results['ann_detail_info']['bboxes'][insta_idx]
                ymin, ymax, xmin, xmax = [int(round(x)) for x in (ymin, ymax, xmin, xmax)]

                cur_bbox_img = results['img'][ymin:ymax, xmin:xmax]
                cur_part_bbox_img = cur_bbox_img * part_mask[..., None]

                img_for_affine = np.zeros_like(results['img'])
                img_for_affine[ymin:ymax, xmin:xmax] = cur_part_bbox_img
                mask_for_affine = np.zeros(results['img'].shape[0:2])
                mask_for_affine[ymin:ymax, xmin:xmax] = part_mask

                # transform anns
                for i in PART_TO_KP_IDS[body_part]:
                    joints[0][insta_idx, i, 0:2] = self._affine_transform(joints[0][insta_idx, i, 0:2],
                                                                          trans_dict[(insta_idx, body_part)])

                # *********  affine subpart **********
                if body_part in PART_TO_SUBPART and PART_TO_SUBPART[body_part] in affine_instance_mask[insta_idx]:
                    subpart = PART_TO_SUBPART[body_part]
                    subpart_mask = affine_instance_mask[insta_idx][subpart]

                    subpart_bbox_img = cur_bbox_img * subpart_mask[..., None]
                    subpart_img_for_affine = np.zeros_like(results['img'])
                    subpart_img_for_affine[ymin:ymax, xmin:xmax] = subpart_bbox_img

                    subpart_mask_for_affine = np.zeros(results['img'].shape[0:2])
                    subpart_mask_for_affine[ymin:ymax, xmin:xmax] = subpart_mask
                    mask_for_affine = mask_for_affine - subpart_mask_for_affine

                    subpart_affined_whole_img = cv2.warpAffine(
                        subpart_img_for_affine,
                        trans_dict[(insta_idx, body_part)],
                        (subpart_img_for_affine.shape[1], subpart_img_for_affine.shape[0]),
                        flags=cv2.INTER_LINEAR
                    )
                    subpart_affined_whole_mask = cv2.warpAffine(
                        subpart_mask_for_affine,
                        trans_dict[(insta_idx, body_part)],
                        (subpart_mask_for_affine.shape[1], subpart_mask_for_affine.shape[0]),
                        flags=cv2.INTER_LINEAR
                    )
                    subpart_affined_whole_img = cv2.warpAffine(
                        subpart_affined_whole_img,
                        trans_dict[(insta_idx, subpart)],
                        (subpart_img_for_affine.shape[1], subpart_img_for_affine.shape[0]),
                        flags=cv2.INTER_LINEAR
                    )
                    subpart_affined_whole_mask = cv2.warpAffine(
                        subpart_affined_whole_mask,
                        trans_dict[(insta_idx, subpart)],
                        (subpart_mask_for_affine.shape[1], subpart_mask_for_affine.shape[0]),
                        flags=cv2.INTER_LINEAR
                    )
                    subpart_affined_whole_mask = subpart_affined_whole_mask[..., None]
                    ret_img = subpart_affined_whole_img * subpart_affined_whole_mask + ret_img * (
                            1 - subpart_affined_whole_mask)
                    # transform anns
                    for i in PART_TO_KP_IDS[subpart]:
                        joints[0][insta_idx, i, 0:2] = self._affine_transform(joints[0][insta_idx, i, 0:2],
                                                                              trans_dict[(insta_idx, subpart)])

                # *********  affine cur part **********
                cur_affined_whole_img = cv2.warpAffine(
                    img_for_affine,
                    trans_dict[(insta_idx, body_part)],
                    (img_for_affine.shape[1], img_for_affine.shape[0]),
                    flags=cv2.INTER_LINEAR
                )
                cur_affined_whole_mask = cv2.warpAffine(
                    mask_for_affine,
                    trans_dict[(insta_idx, body_part)],
                    (mask_for_affine.shape[1], mask_for_affine.shape[0]),
                    flags=cv2.INTER_LINEAR
                )
                cur_affined_whole_mask = cur_affined_whole_mask[..., None]
                ret_img = cur_affined_whole_img * cur_affined_whole_mask + ret_img * (1 - cur_affined_whole_mask)

                if body_part in PART_TO_SUBPART and PART_TO_SUBPART[body_part] in affine_instance_mask[insta_idx]:
                    affine_instance_mask[insta_idx][PART_TO_SUBPART[body_part]] = None

        return ret_img, joints

    def _transform_img_and_anno_topdown(self, affine_instance_mask, results):
        """transform the image and annotations for top-down methods"""
        x, y, w, h = results['bbox']
        joints_3d = results['joints_3d']

        # ********************** Erase original limbs ***********************
        part_mask_sum = np.zeros(results['img'].shape[0:2])
        xmin, ymin, xmax, ymax = int(x), int(y), int(x) + int(round(w)), int(y) + int(round(h))
        ymin, ymax, xmin, xmax = [int(x) for x in (ymin, ymax, xmin, xmax)]
        bbox_img = results['img'][ymin:ymax, xmin:xmax]

        for body_part, part_mask in affine_instance_mask.items():
            part_mask_sum[ymin:ymax, xmin:xmax] += part_mask
        ret_img = cv2.inpaint(results['img'], np.uint8(part_mask_sum), 5, cv2.INPAINT_NS)

        # ********************** Candidate Pose Pool ***********************
        if self.T > 1:
            trans_list = []
            affined_joints_3d = []
            for idx in range(self.T):
                cur_e = 0
                while cur_e < self.E:
                    cur_trans_dict = OrderedDict()
                    cur_joints_3d = np.copy(joints_3d[:, 0:2])
                    for body_part, part_mask in affine_instance_mask.items():
                        cur_trans = self._get_affine_transform(body_part, results)
                        cur_trans_dict[body_part] = cur_trans
                        for j in PART_TO_KP_IDS[body_part]:
                            cur_joints_3d[j] = self._affine_transform(cur_joints_3d[j], cur_trans)
                    cur_e = self.D(results['img'], cur_trans_dict, cur_joints_3d)
                affined_joints_3d.append(cur_joints_3d)
                trans_list.append(cur_trans_dict)

            # predict the probabilities of belonging to each components, then calculate the weighted sum of probabilities
            affined_joints_3d = np.stack(affined_joints_3d)  # (T, 17, 2)
            predict_weights_sum = np.zeros(self.T)

            normal_affined_joints_3d = self.pose_normalize(affined_joints_3d[:, KP_IDS],
                                                           bboxes=results['ann_detail_info']['bboxes'][0])
            cur_kps = normal_affined_joints_3d.reshape(-1, len(KP_IDS) * 2)
            predict_proba = results['gmms'].predict_proba(cur_kps)  # (b, n_components)
            predict_weights_sum += (results['gmms'].weights_ * predict_proba).sum(1)

            min_idx = np.argmin(predict_weights_sum)
            trans_dict = trans_list[min_idx]
        else:
            trans_dict = OrderedDict()
            for body_part, part_mask in affine_instance_mask.items():
                trans_dict[body_part] = self._get_affine_transform(body_part, results)

        # ******************** Start Transformation *******************
        for body_part in list(affine_instance_mask.keys()):

            part_mask = affine_instance_mask[body_part]
            if part_mask is None:
                continue
            cur_part_bbox_img = bbox_img * part_mask[..., None]
            img_for_affine = np.zeros_like(results['img'])
            img_for_affine[ymin:ymax, xmin:xmax] = cur_part_bbox_img

            mask_for_affine = np.zeros(results['img'].shape[0:2])
            mask_for_affine[ymin:ymax, xmin:xmax] = part_mask

            # transform anns
            for i in PART_TO_KP_IDS[body_part]:
                joints_3d[i, 0:2] = self._affine_transform(joints_3d[i, 0:2], trans_dict[body_part])

            # *********  affine subpart **********
            if body_part in PART_TO_SUBPART and PART_TO_SUBPART[body_part] in affine_instance_mask:
                subpart = PART_TO_SUBPART[body_part]
                subpart_mask = affine_instance_mask[subpart]

                subpart_bbox_img = bbox_img * subpart_mask[..., None]
                subpart_img_for_affine = np.zeros_like(results['img'])
                subpart_img_for_affine[ymin:ymax, xmin:xmax] = subpart_bbox_img

                subpart_mask_for_affine = np.zeros(results['img'].shape[0:2])
                subpart_mask_for_affine[ymin:ymax, xmin:xmax] = subpart_mask
                mask_for_affine = mask_for_affine - subpart_mask_for_affine

                subpart_affined_whole_img = cv2.warpAffine(
                    subpart_img_for_affine,
                    trans_dict[body_part],
                    (subpart_img_for_affine.shape[1], subpart_img_for_affine.shape[0]),
                    flags=cv2.INTER_LINEAR
                )
                subpart_affined_whole_mask = cv2.warpAffine(
                    subpart_mask_for_affine,
                    trans_dict[body_part],
                    (subpart_mask_for_affine.shape[1], subpart_mask_for_affine.shape[0]),
                    flags=cv2.INTER_LINEAR
                )
                subpart_affined_whole_img = cv2.warpAffine(
                    subpart_affined_whole_img,
                    trans_dict[subpart],
                    (subpart_img_for_affine.shape[1], subpart_img_for_affine.shape[0]),
                    flags=cv2.INTER_LINEAR
                )
                subpart_affined_whole_mask = cv2.warpAffine(
                    subpart_affined_whole_mask,
                    trans_dict[subpart],
                    (subpart_mask_for_affine.shape[1], subpart_mask_for_affine.shape[0]),
                    flags=cv2.INTER_LINEAR
                )
                subpart_affined_whole_mask = subpart_affined_whole_mask[..., None]
                ret_img = subpart_affined_whole_img * subpart_affined_whole_mask + ret_img * (
                        1 - subpart_affined_whole_mask)
                # transform anns
                for i in PART_TO_KP_IDS[subpart]:
                    joints_3d[i, 0:2] = self._affine_transform(joints_3d[i, 0:2], trans_dict[subpart])

            # *****************  affine cur part  ******************
            cur_affined_whole_img = cv2.warpAffine(
                img_for_affine,
                trans_dict[body_part],
                (img_for_affine.shape[1], img_for_affine.shape[0]),
                flags=cv2.INTER_LINEAR
            )
            cur_affined_whole_mask = cv2.warpAffine(
                mask_for_affine,
                trans_dict[body_part],
                (mask_for_affine.shape[1], mask_for_affine.shape[0]),
                flags=cv2.INTER_LINEAR
            )

            cur_affined_whole_mask = cur_affined_whole_mask[..., None]
            ret_img = cur_affined_whole_img * cur_affined_whole_mask + ret_img * (1 - cur_affined_whole_mask)

            if body_part in PART_TO_SUBPART and PART_TO_SUBPART[body_part] in affine_instance_mask:
                affine_instance_mask[PART_TO_SUBPART[body_part]] = None

        return ret_img, joints_3d

    def __call__(self, results):
        """Apply PoseTrans on a given training sample

        Args:
            results (dict): a dict including meta data for an training sample
                'joints_3d / joints': the ground truth keypoints of the training sample
                                      (joints_3d for top-down method with single instance,
                                      joints for bottom-up method with multiple instances.)
                'img': the input image
                'ann_detail_info' (dict): a dict contains more detail information
                    'bbox': the bounding boxes of the instances
                    'dp_masks': the human parsing results obtained from DensePose model
                'scale': the scale (h, w) of the instance
                'gmms': the fitted GMM used in PCM

        Returns: The transformed results dict

        """
        if 'joints_3d' in results:  # Top Down
            affine_instance_mask = OrderedDict()
            for body_part, aug_probabilities in zip(self.body_parts, self.aug_probabilities):
                if np.random.choice([0, 1], p=[1 - aug_probabilities, aug_probabilities]):
                    part_mask = self._is_exist(body_part, results, 0)
                    if not (part_mask is False):
                        affine_instance_mask[body_part] = part_mask

            if affine_instance_mask:
                new_img, new_joints_3d = self._transform_img_and_anno_topdown(affine_instance_mask, results)
                results['img'] = new_img
                results['joints_3d'] = new_joints_3d

        elif 'joints' in results:  # Bottom Up
            insta_num = len(results['ann_detail_info']['bboxes'])
            affine_instance_mask = OrderedDict()
            for insta_idx in range(insta_num):
                for body_part, aug_probabilities in zip(self.body_parts, self.aug_probabilities):
                    if np.random.choice([0, 1], p=[1 - aug_probabilities, aug_probabilities]):
                        part_mask = self._is_exist(body_part, results, insta_idx)
                        if not (part_mask is False):
                            if insta_idx not in affine_instance_mask:
                                affine_instance_mask[insta_idx] = OrderedDict()
                            affine_instance_mask[insta_idx][body_part] = part_mask

            if affine_instance_mask:
                img, new_joints = self._transform_img_and_anno_bottomup(affine_instance_mask, results)
                results['img'] = img
                results['joints'] = new_joints

        return results

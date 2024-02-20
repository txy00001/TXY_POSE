
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
from rtm_det_pose_track.utils.downloads import safe_download
from pathlib import Path
import numpy as np
from rtm_det_pose_track.utils.core import Detections

base = "rtm"


class RTMDet:
    """RTMDet model (s, m, l) to detect only person (class 0)"""

    def __init__(self, type: str = "l", device: str = "cpu", conf_thres: float = 0.3):
        model_cfg = r'C:\Users\ASUS\Desktop\CastPose\rtm_det_pose_track\conf\rtmdet-l.py'
        onnx_file = Path(r'/onnx/rtmdet-l_640.onnx')

        deploy_cfg = r'C:\Users\ASUS\Desktop\CastPose\mmdeploy\config\mmpose\pose-detection_onnxruntime_static.py'

        deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

        # build task and backend model
        self.task_processor = build_task_processor(model_cfg, deploy_cfg, device)
        self.model = self.task_processor.build_backend_model([onnx_file])

        # process input image
        self.input_shape = get_input_shape(deploy_cfg)

        self.conf_thres = conf_thres

    def __call__(self, im):
        """Return List of xyxy coordinates based on im frame (cv2.imread)
        im -> (h, w, c)
        return -> [[x, y, x, y], [x, y, x, y], ...] -> two persons detected
        """

        model_inputs, _ = self.task_processor.create_input(im, self.input_shape)
        result = self.model.test_step(model_inputs)

        pred_instances = result[0].pred_instances

        # 筛选置信度
        pred_instances = pred_instances[pred_instances.scores > self.conf_thres]

        # 获取类别
        pred_instances = pred_instances[pred_instances.labels == 0].cpu().numpy()

        result = Detections(
            xyxy=pred_instances.bboxes,
            confidence=pred_instances.scores,
            labels=pred_instances.labels,
        )

        return result

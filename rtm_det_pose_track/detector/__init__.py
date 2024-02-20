
from rtm_det_pose_track.detector.rtmdet import RTMDet
from rtm_det_pose_track.detector.yolox import YOLOX

DET_MAP = {"rtmdet": RTMDet,"yolox": YOLOX }


def get_detector(model: str = "rtmdet-l", *args, **kwargs):
    if model.startswith("yolox"):
        return DET_MAP["yolox"]("", *args, **kwargs)

    type, size = model.split("-")

    return DET_MAP[type](size, *args, **kwargs)

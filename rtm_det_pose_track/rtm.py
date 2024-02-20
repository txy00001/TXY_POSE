"""Main class to perform inference using RTMDet and RTMPose (ONNX)"""

import cv2
import csv
from pathlib import Path
import numpy as np
import supervision as sv

from typing import List, Union, Generator, Literal

from rtm_det_pose_track.loader.build import load_inference_source
from rtm_det_pose_track.detector import get_detector
from rtm_det_pose_track.rtmpose import RTMPose

from rtm_det_pose_track.trackers import Tracker, TRACKER_MAP
from rtm_det_pose_track.types import DETECTOR_TYPES, POSE_ESTIMATOR_TYPES, TRACKER_TYPES, DEVICE_TYPES

from rtm_det_pose_track.utils.core import Detections
from rtm_det_pose_track.utils.plotting import Annotator
from rtm_det_pose_track.utils.roi import select_roi
from rtm_det_pose_track.utils.checks import check_imshow
from rtm_det_pose_track.utils.torch_utils import smart_inference_mode
from rtm_det_pose_track.utils.ops import xyxy2xyxyxyxy
from rtm_det_pose_track.utils import (
    LOGGER,
    MACOS,
    WINDOWS,
    colorstr,
    ops,
    get_time,
)

from pydantic import BaseModel


class Result(BaseModel):
    im: np.ndarray  # shape -> (h, w, c)
    kpts: List  # shape -> (number of humans, 17, 2)
    bboxes: List  # shape -> (number of humans, 4)
    speed: dict  # {'bboxes': ... ms, 'kpts': ... ms} -> used to record the milliseconds of the inference time

    save_dirs: str  # save directory
    name: str  # file name

    class Config:
        arbitrary_types_allowed = True


class RTM:
    """"""

    def __init__(
        self,
        det: DETECTOR_TYPES = "rtmdet-m",
        pose: POSE_ESTIMATOR_TYPES = "rtmpose-m",
        tracker: TRACKER_TYPES = "bytetrack",
        device: DEVICE_TYPES = "cpu",
        annotator=Annotator(),
    ) -> None:
        self.det = self.setup_detector(det, device)
        self.rtmpose = RTMPose(pose.split("-")[1], device=device)
        self.annotator = annotator
        self.box_annotator = sv.BoxAnnotator()

        self.tracker_type = tracker

        if tracker not in TRACKER_MAP.keys():
            self.tracker_type = None

        self.dataset = None
        self.vid_path, self.vid_writer = None, None

    def setup_source(self, source, imgsz=640, vid_stride=1) -> None:
        self.dataset = load_inference_source(
            source=source, imgsz=imgsz, vid_stride=vid_stride
        )
        self.source_type = self.dataset.source_type
        self.vid_path, self.vid_writer = [None] * self.dataset.bs, [
            None
        ] * self.dataset.bs

    def setup_tracker(self) -> None:
        self.tracker = Tracker(self.tracker_type).tracker

    def setup_detector(self, det, device):
        return get_detector(
            det,
            device=device,
        )

    def save_preds(self, im0, vid_cap, idx, save_path) -> None:
        """Save video predictions as mp4 at specified path."""
        # Save imgs
        if self.dataset.mode == "image":
            cv2.imwrite(save_path, im0)
        else:  # 'video' or 'stream'
            if self.vid_path[idx] != save_path:  # new video
                self.vid_path[idx] = save_path
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = int(
                        vid_cap.get(cv2.CAP_PROP_FPS)
                    )  # integer required, floats produce error in MP4 codec
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                suffix = ".mp4" if MACOS else ".avi" if WINDOWS else ".avi"
                fourcc = "avc1" if MACOS else "WMV2" if WINDOWS else "MJPG"
                save_path = str(Path(save_path).with_suffix(suffix))
                self.vid_writer[idx] = cv2.VideoWriter(
                    save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h)
                )
            self.vid_writer[idx].write(im0)

    def save_csv(self, path, data):
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(data)

    def get_labels(self, detections: Detections):
        mp = {0: "person"}
        return np.array(
            [
                f"{mp[label]} {id} {score:.2f}"
                for score, label, id in zip(
                    detections.confidence, detections.labels, detections.track_id
                )
            ]
        )

    def setup_roi(self, im, win: str = "roi", type: Literal["rect"] = "rect") -> List:
        w, h, _ = im.shape
        roi_zones, roi_zones_color = select_roi(im, win, type=type)
        roi_zones = xyxy2xyxyxyxy(roi_zones)
        zones = [
            (
                sv.PolygonZone(
                    polygon=zone,
                    frame_resolution_wh=(w, h),
                    triggering_position=sv.Position.CENTER,
                ),
                roi_zones_color[idx],
            )
            for idx, zone in enumerate(roi_zones)
        ]
        return zones

    @smart_inference_mode
    def stream_inference(
        self,
        source,
        # interest point
        timer: List = [],
        poi: Literal["point", "box", "text", ""] = "",
        roi: Literal["rect", ""] = "",
        # panels
        show=True,
        plot=True,
        plot_bboxes=True,
        save=False,
        save_dirs="",
        verbose=True,
    ) -> Result:
        if show:
            check_imshow(warn=True)

        if save:
            save_dirs = Path(get_time() if not save_dirs else save_dirs)
            save_dirs.mkdir(exist_ok=True)

        profilers = (ops.Profile(), ops.Profile(), ops.Profile())  # count the time
        self.setup_source(source)

        if self.tracker_type:
            self.setup_tracker()

        current_source, index = None, 0

        for _, batch in enumerate(self.dataset):
            path, im0s, vid_cap, s = batch
            is_image = s.startswith("image")

            p, im, im0 = Path(path[0]), im0s[0], im0s[0].copy()

            index += 1

            # reset tracker when source changed
            if current_source is None:
                current_source = p
                if roi:
                    zones = self.setup_roi(im)
            elif current_source != p:
                if roi:
                    zones = self.setup_roi(im)
                if self.tracker_type:
                    self.setup_tracker()
                current_source = p
                index = 1

            # multi object detection (detect only person)
            with profilers[0]:
                detections: Detections = self.det(im)

            # multi object tracking (adjust bounding boxes)
            with profilers[1]:
                if self.tracker_type:
                    detections: Detections = self.tracker.update(
                        bboxes=detections.xyxy,
                        confidence=detections.confidence,
                        labels=detections.labels,
                    )
                    track_id = detections.track_id
                else:
                    track_id = np.array([""] * len(detections.xyxy))
                    detections.track_id = track_id

            # filter bboxes based on roi
            if roi and zones and detections:
                masks = set()
                for zone, color in zones:
                    mask = zone.trigger(detections=detections)
                    masks.update(np.where(mask)[0].tolist())
                masks = np.array(
                    [True if i in masks else False for i in range(len(detections.xyxy))]
                )

            # pose estimation (detect 17 keypoints based on the bounding boxes)
            with profilers[2]:
                if detections:
                    kpts = self.rtmpose(im, bboxes=detections.xyxy)

            if plot:
                labels = self.get_labels(detections)

                if roi and zones:
                    for zone, color in zones:
                        im = sv.draw_polygon(im, zone.polygon, color)

                    xyxy, labels, track_id, kpts = (
                        detections.xyxy[masks],
                        labels[masks],
                        track_id[masks],
                        kpts[masks],
                    )
                    detections = Detections(xyxy=xyxy, labels=labels, track_id=track_id)
                    if plot_bboxes:
                        self.box_annotator.annotate(im, detections, labels=labels)
                    self.annotator.draw_kpts(im, kpts)
                    self.annotator.draw_skeletons(im, kpts)
                else:
                    if plot_bboxes:
                        self.box_annotator.annotate(im, detections, labels=labels)
                    self.annotator.draw_kpts(im, kpts)
                    self.annotator.draw_skeletons(im, kpts)

            result = Result(
                im=im,
                kpts=kpts,
                bboxes=detections.xyxy,  # detections.xyxy,
                speed={
                    "bboxes": profilers[0].dt * 1e3 / 1,
                    "track": profilers[1].dt * 1e3 / 1,
                    "kpts": profilers[2].dt * 1e3 / 1,
                },
                save_dirs=str(save_dirs) if save else "",
                name=p.name,
            )

            yield result

            if show:
                if is_image:
                    cv2.imshow(result.name, result.im)
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord("q"):
                        cv2.destroyWindow(result.name)
                        continue
                else:
                    cv2.imshow(result.name, result.im)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == 32:
                        zones = self.setup_roi(im0)

            if verbose:
                LOGGER.info(
                    f"{s}Detected {len(detections.xyxy)} person(s) ({colorstr('green', f'{profilers[0].dt * 1E3:.1f}ms')} | {colorstr('green', f'{profilers[1].dt * 1E3:.1f}ms')} | {colorstr('green', f'{profilers[2].dt * 1E3:.1f}ms')})"
                )

            if save:
                self.save_preds(im, vid_cap, 0, str(save_dirs / p.name))
                self.save_csv(
                    str(save_dirs / p.with_suffix(".csv").name),
                    [
                        str(index),
                        str([{i: kpt} for i, kpt in zip(detections.track_id, kpts)]),
                    ],
                )

        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
            self.vid_writer[-1].release()

        cv2.destroyAllWindows()

    def __call__(
        self,
        source,
        stream=False,
        # interest point
        timer: List = [],
        poi: Literal["point", "box", "text", ""] = "",
        roi: Literal["rect", ""] = "",
        # panels
        show=True,
        plot=True,
        plot_bboxes=True,
        save=False,
        save_dirs="",
        verbose=True,
    ) -> Union[List[Result], Generator[Result, None, None]]:
        if stream:
            return self.stream_inference(
                source,
                timer=timer,
                poi=poi,
                roi=roi,
                show=show,
                plot=plot,
                plot_bboxes=plot_bboxes,
                save=save,
                save_dirs=save_dirs,
                verbose=verbose,
            )
        else:
            return list(
                self.stream_inference(
                    source,
                    timer=timer,
                    poi=poi,
                    roi=roi,
                    show=show,
                    plot=plot,
                    plot_bboxes=plot_bboxes,
                    save=save,
                    save_dirs=save_dirs,
                    verbose=verbose,
                )
            )

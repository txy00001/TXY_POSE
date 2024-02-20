import os
from abc import ABCMeta, abstractmethod
from typing import Any

import cv2
import numpy as np

from lights.tools.file import download_checkpoint

CAST_LIGHTS_SETTINGS = {
    'opencv': {
        'cpu': (cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU),

        'cuda': (cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA)
    },
    'onnxruntime': {
        'cpu': 'CPUExecutionProvider',
        'cuda': 'CUDAExecutionProvider'
    },
}


class BaseTool(metaclass=ABCMeta):

    def __init__(self,
                 onnx_model: str = None,
                 model_input_size: tuple = None,
                 mean: tuple = None,
                 std: tuple = None,
                 backend: str = 'opencv',
                 device: str = 'gpu'):

        if not os.path.exists(onnx_model):
            onnx_model = download_checkpoint(onnx_model)

        if backend == 'opencv':###opencv后端
            try:
                providers = CAST_LIGHTS_SETTINGS[backend][device]

                session = cv2.dnn.readNetFromONNX(onnx_model)
                session.setPreferableBackend(providers[0])
                session.setPreferableTarget(providers[1])
                self.session = session
            except Exception:
                raise RuntimeError(
                    'This model is not supported by OpenCV'
                    ' backend, please use `pip install'
                    ' onnxruntime` or `pip install'
                    ' onnxruntime-gpu` to install onnxruntime'
                    ' backend. Then specify `backend=onnxruntime`.')  # noqa

        elif backend == 'onnxruntime':
            import onnxruntime as ort
            providers = CAST_LIGHTS_SETTINGS[backend][device]

            self.session = ort.InferenceSession(path_or_bytes=onnx_model,
                                                providers=[providers])
        

        else:
            raise NotImplementedError

        print(f'load {onnx_model} with {backend} backend')

        self.onnx_model = onnx_model
        self.model_input_size = model_input_size
        self.mean = mean
        self.std = std
        self.backend = backend
        self.device = device

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Implement the actual function here."""
        raise NotImplementedError

    def inference(self, img: np.ndarray):
       
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        input = img[None, :, :, :]

        # run model
        if self.backend == 'opencv':
            outNames = self.session.getUnconnectedOutLayersNames()
            self.session.setInput(input)
            outputs = self.session.forward(outNames)
        elif self.backend == 'onnxruntime':
            sess_input = {self.session.get_inputs()[0].name: input}
            sess_output = []
            for out in self.session.get_outputs():
                sess_output.append(out.name)

            outputs = self.session.run(sess_output, sess_input)
        elif self.backend == 'openvino':
            results = self.compiled_model(input)
            output0 = results[self.output_layer0]
            output1 = results[self.output_layer1]
            if output0.ndim == output1.ndim:
                outputs = [output0, output1]
            else:
                outputs = [output0]

        return outputs
import onnxmltools
from onnxmltools.utils.float16_converter import convert_float_to_float16
# from winmltools.utils import convert_
# from winmltools.utils import load_model, save_model
for modelname in [r"C:\Users\Administrator\Desktop\mmpose_mydw\data\ckpts\det_qx.onnx"]:
    # Update the input name and path for your ONNX model
    input_onnx_model = modelname
    # Change this path to the output name and path for your float16 ONNX model
    output_onnx_model = modelname[:-5] + '_bf16.onnx'
    print(output_onnx_model)
    # Load your model
    onnx_model = onnxmltools.utils.load_model(input_onnx_model)
    # Convert tensor float type from your input ONNX model to tensor float16
    onnx_model = convert_float_to_float16(onnx_model)
    # Save as protobuf
    onnxmltools.utils.save_model(onnx_model, output_onnx_model)
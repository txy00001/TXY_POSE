# import onnx
# # from onnxruntime.quantization import QuantType, quantize_dynamic ,quantize_qat
# # # 模型路径 model_fp32 = 'models/MobileNetV1_infer.onnx'
# # for modelname in ["nanodet-plus-m_416.onnx","picodet_m_416_coco - 副本.onnx"]:
# #     output_onnx_model = modelname[:-5]+'_int8.onnx'
# #     print (modelname)
# #     quantize_qat( model_input=modelname,model_output=output_onnx_model,weight_type=QuantType.QUInt8)
# #
# # import onnx
# # # from onnxruntime.quantization import quantize_dynamic, QuantType,quantize_qat
# # # for modelname in ["hand_landmark_lite-sim.onnx"]:
# # #    model_fp32 = 'E:\porject\onnx2other\hand_landmark_lite-sim.onnx'
# # #    output_onnx_model = modelname[:-5] + '_int8.onnx'
# # #    quantized_model = quantize_dynamic(model_fp32, output_onnx_model, weight_type=QuantType.QUInt8)
#
# import onnx
# import onnxruntime as ort
# import pytorch_lightning as pl
from onnxruntime.quantization import QuantType, quantize_dynamic,CalibrationDataReader, QuantFormat, quantize_static, QuantType, CalibrationMethod

# 模型路径
model_fp32 = r'C:\Users\Administrator\Desktop\mmpose_mydw\data\ckpts\det_qx.onnx'
model_quant_dynamic =  r'C:\Users\Administrator\Desktop\mmpose_mydw\data\ckpts\det_qx_int8.onnx'

# 动态量化
quantize_dynamic(
    model_input=model_fp32, # 输入模型
    model_output=model_quant_dynamic, # 输出模型
    weight_type=QuantType.QUInt8, # 参数类型 Int8 / UInt8
    optimize_model=True # 是否优化模型
     # 激活类型 Int8 / UInt8

)

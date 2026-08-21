# quantize.py
from onnxruntime.quantization import quantize_dynamic, QuantType

print("Quantizing bge_m3_onnx/model.onnx to INT8...")
quantize_dynamic(
    model_input="bge_m3_onnx/model.onnx",
    model_output="bge_m3_onnx/model_quantized.onnx",
    weight_type=QuantType.QUInt8
)
print("Quantization complete: bge_m3_onnx/model_quantized.onnx created!")
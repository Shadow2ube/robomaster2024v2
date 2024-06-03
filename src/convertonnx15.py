import onnx
import sys
from onnx import version_converter

model_path = sys.argv[1]
original_model = onnx.load(model_path)

converted_model = version_converter.convert_version(original_model, 15)
onnx.save(converted_model, sys.argv[2])

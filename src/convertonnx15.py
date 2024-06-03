import onnx
import sys
from onnx import version_converter

model_path = sys.argv[1]
print('loading model from:', model_path, '...', end='')
original_model = onnx.load_model(model_path)
print('done')

print('converting model...', end='')
converted_model = version_converter.convert_version(original_model, 15)
print('done')

print('saving to: ', sys.argv[2], '...', end='')
onnx.save(converted_model, sys.argv[2])
print('done')

# Based on this: https://stephencowchau.medium.com/stitching-non-max-suppression-nms-to-yolov8n-on-exported-onnx-model-1c625021b22

# * model is the YOLOv8n trained (YOLO class)
# * batch = 1 is important here,
#     even export support dynamic axis using dynamic=True,
#     sometimes it just fail to export
import sys
import onnx
# import onnxsim
import torch
from onnx import TensorProto
from onnx.compose import merge_models
from onnx.tools import update_model_dims
from onnx.version_converter import convert_version
from torch import nn
from ultralytics import YOLO

model = YOLO(sys.argv[1])
onnx_model_path = './model.onnx'

model.export(format='onnx', simplify=True, imgsz=[640, 640], batch=1)

# load the model and manipulate it
onnx_model = onnx.load_model(onnx_model_path)
onnx_fpath = "./best_nms.onnx"

graph = onnx_model.graph

# operation to transpose bbox before pass to NMS node
transpose_bboxes_node = onnx.helper.make_node("Transpose", inputs=["/model.22/Mul_2_output_0"], outputs=["bboxes"],
                                              perm=(0, 2, 1))
graph.node.append(transpose_bboxes_node)

# make constant tensors for nms
score_threshold = onnx.helper.make_tensor("score_threshold", TensorProto.FLOAT, [1], [0.25])
iou_threshold = onnx.helper.make_tensor("iou_threshold", TensorProto.FLOAT, [1], [0.45])
max_output_boxes_per_class = onnx.helper.make_tensor("max_output_boxes_per_class", TensorProto.INT64, [1], [200])

# create the NMS node
inputs = ['bboxes', '/model.22/Sigmoid_output_0', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold', ]
# inputs=['onnx::Concat_458', 'onnx::Concat_459', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold',]
outputs = ["selected_indices"]
nms_node = onnx.helper.make_node(
    'NonMaxSuppression',
    inputs,
    ["selected_indices"],
    # center_point_box=1 is very important, PyTorch model's output is
    #  [x_center, y_center, width, height], but default NMS expect
    #  [x_min, y_min, x_max, y_max]
    center_point_box=1,
)

# add NMS node to the list of graph nodes
graph.node.append(nms_node)

# append to the output (now the outputs would be scores, bboxes, selected_indices)
output_value_info = onnx.helper.make_tensor_value_info("selected_indices", TensorProto.INT64, shape=["num_results", 3])
graph.output.append(output_value_info)

# add to initializers - without this, onnx will not know where these came from, and complain that
# they're neither outputs of other nodes, nor inputs. As initializers, however, they are treated
# as constants needed for the NMS op
graph.initializer.append(score_threshold)
graph.initializer.append(iou_threshold)
graph.initializer.append(max_output_boxes_per_class)

# remove the unused concat node
last_concat_node = [node for node in onnx_model.graph.node if node.name == "/model.22/Concat_5"][0]
onnx_model.graph.node.remove(last_concat_node)

# remove the original output0
output0 = [o for o in onnx_model.graph.output if o.name == "output0"][0]
onnx_model.graph.output.remove(output0)

# output keep for downstream task
graph.output.append([v for v in onnx_model.graph.value_info if v.name == "/model.22/Mul_2_output_0"][0])
graph.output.append([v for v in onnx_model.graph.value_info if v.name == "/model.22/Sigmoid_output_0"][0])

# check that it works and re-save
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, onnx_fpath)

input_dims = {
    "images": ["batch", 3, 640, 640],
}

output_dims = {
    "selected_indices": ["num_results", 3],
    "/model.22/Mul_2_output_0": ["batch", "boxes", "num_anchors"],
    "/model.22/Sigmoid_output_0": ["batch", "classes", "num_anchors"],
}

updated_onnx_model = update_model_dims.update_inputs_outputs_dims(onnx_model, input_dims, output_dims)


class Transform(nn.Module):
    def forward(self, idxTensor, boxes, scores):
        bbox_result = self.gather(boxes, idxTensor)
        score_intermediate_result = self.gather(scores, idxTensor).max(axis=-1)
        score_result = score_intermediate_result.values
        classes_result = score_intermediate_result.indices
        num_dets = torch.tensor(score_result.shape[-1])
        return bbox_result, score_result, classes_result, num_dets

    '''
    Input:
    boxes: [bs=1, 4, 8400]
    indices: [N, 3]
  
    expect output
    '''

    def gather(self, target, idxTensor):
        pick_indices = idxTensor[:, -1:].repeat(1, target.shape[1]).unsqueeze(0)
        return torch.gather(target.permute(0, 2, 1), 1, pick_indices)


torch_boxes = torch.tensor([
    [91.0, 2, 3, 4, 5, 6],
    [11, 12, 13, 14, 15, 16],
    [21, 22, 23, 24, 25, 26],
    [31, 32, 33, 34, 35, 36],
]).unsqueeze(0)

torch_scores = torch.tensor([
    [0.1, 0.82, 0.3, 0.6, 0.55, 0.6],
    [0.9, 0.18, 0.7, 0.4, 0.45, 0.4],
]).unsqueeze(0)

torch_indices = torch.tensor([[0, 0, 0], [0, 0, 2], [0, 0, 1]])

t_model = Transform()

torch.onnx.export(t_model, (torch_indices, torch_boxes, torch_scores), "./NMS_after.onnx",
                  input_names=["selected_indices", "boxes", "scores"],
                  output_names=["det_bboxes", "det_scores", "det_classes", "num_dets"],
                  dynamic_axes={
                      "boxes": {0: "batch", 1: "boxes", 2: "num_anchors"},
                      "scores": {0: "batch", 1: "classes", 2: "num_anchors"},
                      "selected_indices": {0: "num_results"},
                      "det_bboxes": {1: "num_results"},
                      "det_scores": {1: "num_results"},
                      "det_classes": {1: "num_results"},
                  })

nms_postprocess_onnx_model = onnx.load_model("./NMS_after.onnx")
# nms_postprocess_onnx_model_sim, check = onnxsim.simplify(nms_postprocess_onnx_model)
onnx.save(nms_postprocess_onnx_model, "./model_sim.onnx")

combined_onnx_path = "./final.onnx"

target_ir_version = 15
core_model = convert_version(updated_onnx_model, target_ir_version)
# this output is weird, it still say it's version 8, even after convert
print(f"core_model version : {core_model.ir_version}")
onnx.checker.check_model(core_model)
# force to pass the version check, the convert seems success but the ir_version does NOT change
core_model.ir_version = 15

# core_model = updated_onnx_model
# post_process_model = convert_version(nms_postprocess_onnx_model_sim, target_ir_version)
post_process_model = convert_version(nms_postprocess_onnx_model, target_ir_version)
# this output is weird, it still say it's version 7, even after convert
print(f"post_process_model version : {post_process_model.ir_version}")
onnx.checker.check_model(post_process_model)
# force to pass the version check, the convert seems success but the ir_version does NOT change
post_process_model.ir_version = 15

combined_onnx_model = merge_models(core_model, post_process_model, io_map=[
    ('/model.22/Mul_2_output_0', 'boxes'),
    ('/model.22/Sigmoid_output_0', 'scores'),
    ('selected_indices', 'selected_indices')
])

core_model = convert_version(combined_onnx_model, 15)
core_model.ir_version = 15
onnx.save(combined_onnx_model, combined_onnx_path)

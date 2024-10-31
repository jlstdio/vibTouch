"""
.pth to .onnx
"""
import torch
import onnx
from onnxsim import simplify
from modelArch.MultimodalClassifier1D import MultimodalClassifier1D

root = f'./model'
modelName = 'gesture_transformer_epoch409'
pth_path = f'{root}/pth/{modelName}.pth'
onnx_path = f'{root}/onnx/{modelName}.onnx'
onnx_simplified_path = f'{root}/onnx/{modelName}_onnx_simplified_path.onnx'

model = MultimodalClassifier1D()
model.load_state_dict(torch.load(pth_path, map_location='cpu'))
model.eval()

# 모델을 ONNX로 변환
input_acc = torch.randn(1, 3, 12800)
input_audio = torch.randn(1, 1, 12800)
torch.onnx.export(model, (input_acc, input_audio), onnx_path, opset_version=12)

# load your predefined ONNX model
model = onnx.load(onnx_path)

for idx, node in enumerate(model.graph.node):
    # node.name = f"{node.name}_{idx}"
    print(node.name + ' -> ', end='')
    if node.name.startswith('/'):
        node.name = node.name[1:]
    print(node.name)

onnx.save(model, onnx_path)

# convert model
model_simp, check = simplify(model)

assert check, "Simplified ONNX model could not be validated"

# 각 노드의 이름에 접미사를 추가하여 고유하게 변경
for idx, node in enumerate(model_simp.graph.node):
    # node.name = f"{node.name}_{idx}"
    print(node.name + ' -> ', end='')
    if node.name.startswith('/'):
        node.name = node.name[1:]
    print(node.name)

# use model_simp as a standard ONNX model object
onnx.save(model_simp, onnx_simplified_path)

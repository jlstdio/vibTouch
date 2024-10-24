"""
.pth to .onnx
"""
import torch
from model import EAST
import onnx
from onnxsim import simplify

pth_path = "{model_directory_path}/best.pth"
onnx_path = '{target_directory_path}/best.onnx'


model = CustomModel()
model.load_state_dict(torch.load(pth_path, map_location='cpu'))
model.eval()

# 모델을 ONNX로 변환
input_sample = torch.randn(1, 3, 1024, 1024)  # 입력 샘플 생성
torch.onnx.export(model, input_sample, onnx_path, opset_version=11)

# load your predefined ONNX model
model = onnx.load(onnx_path)

# convert model
model_simp, check = simplify(model)

assert check, "Simplified ONNX model could not be validated"

# use model_simp as a standard ONNX model object
onnx.save(model_simp, "best_simplified.onnx")
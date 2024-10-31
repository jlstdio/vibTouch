import torch
import numpy as np
from modelArch.MultimodalClassifier1D import MultimodalClassifier1D

root = f'./model'
modelName = 'gesture_transformer_epoch409'

# Load the .pth file
pth_path = f'{root}/pth/{modelName}.pth'
model = MultimodalClassifier1D()
model.load_state_dict(torch.load(pth_path, map_location='cpu'))
model.eval()  # Set the model to evaluation mode

# Prepare input data
input_acc = torch.randn(1, 3, 12800)
input_audio = torch.randn(1, 1, 12800)

# Make predictions
with torch.no_grad():
    output = model(input_acc, input_audio)

# Print the output
print(output[0].type())
print(torch.max(output[0]), torch.min(output[0]))

"""
tensorflow model validation
"""
import tensorflow as tf

# FrozenGraph 모델 경로
frozen_graph_path = f'{root}/pb/{modelName}_pb.pb'

# TensorFlow 2.x에서 FrozenGraph 로드
with tf.io.gfile.GFile(frozen_graph_path, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

# 로드한 모델을 기반으로 TensorFlow 그래프 생성
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")

# 모델 실행
with tf.compat.v1.Session(graph=graph) as sess:
    # 입출력 노드 정의
    input_tensor1 = graph.get_tensor_by_name("input.1:0")  # 첫 번째 입력 텐서
    input_tensor2 = graph.get_tensor_by_name("input.19:0")  # 두 번째 입력 텐서
    output_tensor = graph.get_tensor_by_name("82:0")  # 출력 텐서의 이름

    # 입력 데이터
    input_data1 = np.full((1, 3, 12800), 0.5).astype(np.float32)  # 첫 번째 입력 데이터
    input_data2 = np.full((1, 1, 12800), 0.5).astype(np.float32)  # 두 번째 입력 데이터

    # 모델 실행
    output_data = sess.run(output_tensor, feed_dict={input_tensor1: input_data1, input_tensor2: input_data2})

    print(np.max(output_data), np.min(output_data))

"""
tflite model validation
"""
# TFLite 모델 파일 경로
tflite_model_path = f"{root}/tflite/{modelName}_tflite.tflite"

# TFLite 모델 로드
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# 입력 텐서와 출력 텐서 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 입력 데이터 생성
input_data1 = np.full((1, 3, 12800), 0.5).astype(np.float32)  # 첫 번째 입력 데이터
input_data2 = np.full((1, 1, 12800), 0.5).astype(np.float32)  # 두 번째 입력 데이터

# 입력 데이터 설정
interpreter.set_tensor(input_details[0]['index'], input_data1)
interpreter.set_tensor(input_details[1]['index'], input_data2)

# 모델 실행
interpreter.invoke()

# 출력 데이터 가져오기
output_data = interpreter.get_tensor(output_details[0]['index'])

print(np.max(output_data), np.min(output_data))

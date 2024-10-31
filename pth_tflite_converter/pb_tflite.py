import tensorflow as tf

root = f'./model'
modelName = 'gesture_transformer_epoch409'
pb_model_path = f'{root}/pb/{modelName}_pb.pb'
tflite_model_path = f"{root}/tflite/{modelName}_tflite.tflite"

# TensorFlow Lite 포맷으로 모델 변환
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(pb_model_path, #TensorFlow freezegraph .pb model file
                                                      input_arrays=['input.1', 'input.19'], # name of input arrays as defined in torch.onnx.export function before.
                                                      output_arrays=['82']  # name of output arrays defined in torch.onnx.export function before.
                                                      )

converter.optimizations = [tf.lite.Optimize.DEFAULT]	# 최적화
# converter.optimizations = []	# 최적화
tflite_model = converter.convert()	# tflite로 변환

# TensorFlow Lite 모델 저장
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
import onnx
from onnx_tf.backend import prepare


root = f'./model'
modelName = 'gesture_transformer_epoch409'
onnx_original_path = f'{root}/onnx/{modelName}.onnx'
onnx_simplified_path = f'{root}/onnx/{modelName}_onnx_simplified_path.onnx'

pb_model_path = f'{root}/pb/{modelName}_pb.pb'

onnx_model = onnx.load(onnx_simplified_path)

tf_rep = prepare(onnx_model)
tf_rep.export_graph(pb_model_path)
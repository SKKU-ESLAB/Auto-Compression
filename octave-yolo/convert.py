import onnx
import os
from onnx_tf.backend import prepare
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5, help="alpha")
args = parser.parse_args()
print(args)

onnx_path = "/home/sangjun/octyolo/yolov3-pytorch/"+str(args.alpha)+".onnx"
onnx_model = onnx.load(onnx_path)
tf_rep = prepare(onnx_model)

pb_path = "./"+str(args.alpha)+".pb"
"""tf_rep.export_graph(pb_path)

assert os.path.exists(pb_path)
print(".pb model converted successfully.")"""

input_nodes = tf_rep.inputs
output_nodes = tf_rep.outputs

print("The names of the input nodes are: {}".format(input_nodes))
print("The names of the output nodes are: {}".format(output_nodes))

converter = tf.lite.TFLiteConverter.from_saved_model(pb_path)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS, # enable TensorFlow ops.
  tf.strided_slice
]
tflite_rep = converter.convert()
tflite_path = "./"+str(args.alpha)+".tflite"
open(tflite_path, "wb").write(tflite_rep)

assert os.path.exists(tflite_path)
print(".tflite model converted successfully.")
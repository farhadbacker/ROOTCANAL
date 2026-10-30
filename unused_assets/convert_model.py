import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import tensorflowjs as tfjs
import os

# Load the ONNX model
print("Loading ONNX model...")
onnx_model = onnx.load('public/dental_yolo.onnx')
print("ONNX model loaded successfully")

# Convert ONNX to TensorFlow
print("Converting to TensorFlow...")
tf_rep = prepare(onnx_model)
print("Conversion completed")

# Get the TensorFlow model
tf_model = tf_rep.tf_module
print("TensorFlow model extracted")

# Create output directory
os.makedirs('public/models/tfjs', exist_ok=True)

# Convert to TensorFlow.js format
print("Converting to TensorFlow.js...")
tfjs.converters.save_keras_model(tf_model, 'public/models/tfjs')
print("TensorFlow.js model saved to public/models/tfjs/")
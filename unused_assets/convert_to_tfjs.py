import onnxruntime as ort
import tensorflow as tf
import tf2onnx
import os
import numpy as np

def convert_onnx_to_tfjs(onnx_path, tfjs_output_dir):
    """
    Convert ONNX model to TensorFlow.js format
    """
    print(f"Converting {onnx_path} to TensorFlow.js...")

    # First, convert ONNX to TensorFlow SavedModel
    tf_model_path = onnx_path.replace('.onnx', '_tf')

    # Use tf2onnx to convert ONNX to TensorFlow
    import onnx
    from tf2onnx import convert

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)

    # Convert to TensorFlow
    tf_rep = convert.from_onnx(onnx_model)

    # Save as TensorFlow SavedModel
    tf.saved_model.save(tf_rep, tf_model_path)
    print(f"Saved TensorFlow model to {tf_model_path}")

    # Convert to TensorFlow.js
    os.system(f"tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model {tf_model_path} {tfjs_output_dir}")
    print(f"Converted to TensorFlow.js in {tfjs_output_dir}")

if __name__ == "__main__":
    onnx_path = "public/dental_yolo.onnx"
    tfjs_output_dir = "public/models/tfjs"

    if os.path.exists(onnx_path):
        os.makedirs(tfjs_output_dir, exist_ok=True)
        convert_onnx_to_tfjs(onnx_path, tfjs_output_dir)
    else:
        print(f"ONNX model not found at {onnx_path}")
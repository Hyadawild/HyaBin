"""
Convert CNN-LSTM model from .h5 to ONNX format
Run this ONCE locally, then commit the .onnx file to your repo

Usage:
    python convert_to_onnx.py
"""

import tensorflow as tf
import tf2onnx
import onnx
import numpy as np

print("🔄 Converting CNN-LSTM model to ONNX format...")
print("This may take a few minutes...")

try:
    # Load the trained model
    print("\n1️⃣ Loading model...")
    model = tf.keras.models.load_model('best_cnn_lstm_model.h5')
    print(f"   ✅ Model loaded. Input shape: {model.input_shape}")
    
    # Convert to ONNX
    print("\n2️⃣ Converting to ONNX...")
    spec = (tf.TensorSpec((None, 30, 5), tf.float32, name="input"),)
    output_path = "best_cnn_lstm_model.onnx"
    
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
    
    # Verify ONNX model
    print("\n3️⃣ Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"\n✅ SUCCESS! Model converted to ONNX format!")
    print(f"   Output file: {output_path}")
    print(f"   File size: ~{onnx.load(output_path).__sizeof__() / 1024 / 1024:.1f} MB")
    print(f"\n🚀 Next steps:")
    print(f"   1. Commit {output_path} to your repository")
    print(f"   2. Push the changes to GitHub/GitLab")
    print(f"   3. The app will use ONNX Runtime for inference (no TensorFlow needed!)")
    
except Exception as e:
    print(f"\n❌ Error during conversion: {str(e)}")
    print("\nMake sure you have tf2onnx installed:")
    print("  pip install tf2onnx onnx")
    raise

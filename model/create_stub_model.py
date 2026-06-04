"""
create_stub_model.py

Generates a phishing_model.onnx with the correct input/output signatures
but random (untrained) weights. Use this to verify the Flask backend and
Chrome extension work end-to-end before you have real training data.

The stub will produce random scores — outputs are meaningless for detection.

Usage:
    pip install -r requirements_training.txt
    python create_stub_model.py
    # → writes phishing_model.onnx in model/
"""

import os
import sys

import tensorflow as tf
from tensorflow.keras import layers, Model
import tf2onnx
import onnx

# Must match feature_extractor.py and train_model.py exactly.
URL_MAX_LEN    = 256
DOM_MAX_LEN    = 500
META_DIM       = 20
URL_VOCAB_SIZE = 85
DOM_VOCAB_SIZE = 61

OUT_PATH = os.path.join(os.path.dirname(__file__), 'phishing_model.onnx')


def build_stub() -> Model:
    url_input = layers.Input(shape=(URL_MAX_LEN,), dtype='int32', name='url_input')
    url_emb   = layers.Embedding(URL_VOCAB_SIZE + 1, 32, mask_zero=True)(url_input)
    url_feat  = layers.Bidirectional(layers.LSTM(64))(url_emb)
    url_score = layers.Dense(1, activation='sigmoid', name='url_output')(url_feat)

    dom_input = layers.Input(shape=(DOM_MAX_LEN,), dtype='int32', name='dom_input')
    dom_emb   = layers.Embedding(DOM_VOCAB_SIZE + 1, 16, mask_zero=True)(dom_input)
    dom_conv  = layers.Conv1D(64, kernel_size=5, activation='relu')(dom_emb)
    dom_feat  = layers.GlobalMaxPooling1D()(dom_conv)
    dom_score = layers.Dense(1, activation='sigmoid', name='dom_output')(dom_feat)

    meta_input = layers.Input(shape=(META_DIM,), dtype='float32', name='meta_input')
    meta_h1    = layers.Dense(32, activation='relu')(meta_input)
    meta_feat  = layers.Dense(16, activation='relu')(meta_h1)
    meta_score = layers.Dense(1, activation='sigmoid', name='meta_output')(meta_feat)

    fused   = layers.Concatenate()([url_feat, dom_feat, meta_feat])
    dense   = layers.Dense(32, activation='relu')(fused)
    dropout = layers.Dropout(0.5)(dense)
    output  = layers.Dense(1, activation='sigmoid', name='output')(dropout)

    return Model(
        inputs=[url_input, dom_input, meta_input],
        outputs=[output, url_score, dom_score, meta_score],
    )


if __name__ == '__main__':
    print('[create_stub_model] Building stub model...')
    model = build_stub()

    spec = (
        tf.TensorSpec((None, URL_MAX_LEN), tf.int32,   name='url_input'),
        tf.TensorSpec((None, DOM_MAX_LEN), tf.int32,   name='dom_input'),
        tf.TensorSpec((None, META_DIM),    tf.float32, name='meta_input'),
    )

    print('[create_stub_model] Converting to ONNX...')
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    onnx.save(model_proto, OUT_PATH)

    print(f'[create_stub_model] Stub saved → {OUT_PATH}')
    print('NOTE: This model has random weights. Run train_model.py to train on real data.')

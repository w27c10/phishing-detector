"""
train_model.py — Multi-branch phishing detection model (TensorFlow/Keras → ONNX)

Architecture:
  Branch 1 : Bidirectional LSTM  — character-level URL tokenisation (256 chars)
  Branch 2 : 1-D CNN             — HTML tag-sequence encoding (500 tags)
  Branch 3 : DNN                 — 20 hand-crafted metadata features
  Fusion    : Concatenate → Dense(32, relu) → Dropout(0.5) → Dense(1, sigmoid)

Each branch also has its own sigmoid head so the Flask backend can report
per-modality threat scores for the XAI dashboard.

Dataset CSV format:
  url    : raw URL string
  dom    : raw HTML string (may be empty for URL-only datasets)
  label  : 0 (safe) or 1 (phishing)

Usage:
    pip install -r requirements_training.txt
    python train_model.py dataset.csv
    # → writes phishing_model.onnx to model/
"""

import sys
import os

import numpy as np
import pandas as pd

# ── Model constants (must stay in sync with feature_extractor.py) ──────────────
URL_MAX_LEN   = 256
DOM_MAX_LEN   = 500
META_DIM      = 20
URL_VOCAB_SIZE = 85
DOM_VOCAB_SIZE = 61


def build_model():
    """Return the compiled multi-branch Keras model."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model, regularizers

    l2 = regularizers.l2(1e-4)

    # ── Branch 1: URL Lexical (Bidirectional LSTM) ─────────────────────────────
    url_input = layers.Input(shape=(URL_MAX_LEN,), dtype='int32', name='url_input')
    url_emb   = layers.Embedding(URL_VOCAB_SIZE + 1, 32, mask_zero=True)(url_input)
    url_emb   = layers.SpatialDropout1D(0.1)(url_emb)
    url_feat  = layers.Bidirectional(layers.LSTM(64))(url_emb)
    url_feat  = layers.Dropout(0.2)(url_feat)
    url_score = layers.Dense(1, activation='sigmoid', name='url_output')(url_feat)

    # ── Branch 2: DOM Structure (1-D CNN) ──────────────────────────────────────
    dom_input = layers.Input(shape=(DOM_MAX_LEN,), dtype='int32', name='dom_input')
    dom_emb   = layers.Embedding(DOM_VOCAB_SIZE + 1, 16, mask_zero=True)(dom_input)
    dom_emb   = layers.SpatialDropout1D(0.2)(dom_emb)
    dom_conv  = layers.Conv1D(64, kernel_size=5, activation='relu', kernel_regularizer=l2)(dom_emb)
    dom_feat  = layers.GlobalMaxPooling1D()(dom_conv)
    dom_feat  = layers.Dropout(0.3)(dom_feat)
    dom_score = layers.Dense(1, activation='sigmoid', name='dom_output')(dom_feat)

    # ── Branch 3: Metadata (DNN) ───────────────────────────────────────────────
    meta_input = layers.Input(shape=(META_DIM,), dtype='float32', name='meta_input')
    meta_h1    = layers.Dense(32, activation='relu', kernel_regularizer=l2)(meta_input)
    meta_h1    = layers.Dropout(0.3)(meta_h1)
    meta_feat  = layers.Dense(16, activation='relu', kernel_regularizer=l2)(meta_h1)
    meta_score = layers.Dense(1, activation='sigmoid', name='meta_output')(meta_feat)

    # ── Fusion ─────────────────────────────────────────────────────────────────
    fused   = layers.Concatenate()([url_feat, dom_feat, meta_feat])
    dense   = layers.Dense(32, activation='relu', kernel_regularizer=l2)(fused)
    dropout = layers.Dropout(0.5)(dense)
    output  = layers.Dense(1, activation='sigmoid', name='output')(dropout)

    model = Model(
        inputs=[url_input, dom_input, meta_input],
        outputs=[output, url_score, dom_score, meta_score],
    )

    # Auxiliary branch losses weighted lightly; main output carries the
    # penalised loss. class_weight applied separately in model.fit().
    model.compile(
        optimizer='adam',
        loss={
            'output':      tf.keras.losses.BinaryCrossentropy(),
            'url_output':  tf.keras.losses.BinaryCrossentropy(),
            'dom_output':  tf.keras.losses.BinaryCrossentropy(),
            'meta_output': tf.keras.losses.BinaryCrossentropy(),
        },
        loss_weights={
            'output':      1.0,   # Primary objective
            'url_output':  0.3,   # Auxiliary regularisation
            'dom_output':  0.3,
            'meta_output': 0.3,
        },
        metrics={'output': ['accuracy', tf.keras.metrics.Recall(name='recall')]},
    )
    return model


def load_dataset(csv_path: str):
    """Load CSV and convert rows to feature tensors."""
    # Import here so the file is importable without TF installed.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
    from feature_extractor import (
        extract_url_features,
        extract_dom_features,
        extract_metadata_features,
    )

    if csv_path.endswith('.pkl'):
        df = pd.read_pickle(csv_path)
    else:
        df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
    df['url'] = df['url'].fillna('').astype(str).str.replace('\x00', '', regex=False)
    df['dom'] = df.get('dom', pd.Series([''] * len(df))).fillna('').astype(str).str.replace('\x00', '', regex=False)
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    df = df[df['url'].str.startswith('http')].reset_index(drop=True)

    required = {'url', 'label'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    from tqdm import tqdm
    url_rows, dom_rows, meta_rows, labels = [], [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Extracting features', ncols=80):
        url = str(row['url'])
        dom = str(row.get('dom', ''))
        url_rows.append(extract_url_features(url)[0])
        dom_rows.append(extract_dom_features(dom)[0])
        meta_rows.append(extract_metadata_features(url, dom)[0])
        labels.append(float(row['label']))

    return (
        np.array(url_rows,  dtype=np.int32),
        np.array(dom_rows,  dtype=np.int32),
        np.array(meta_rows, dtype=np.float32),
        np.array(labels,    dtype=np.float32),
    )


def export_to_onnx(model, out_path: str) -> None:
    import tensorflow as tf
    import tf2onnx
    import onnx

    spec = (
        tf.TensorSpec((None, URL_MAX_LEN), tf.int32,   name='url_input'),
        tf.TensorSpec((None, DOM_MAX_LEN), tf.int32,   name='dom_input'),
        tf.TensorSpec((None, META_DIM),    tf.float32, name='meta_input'),
    )
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    onnx.save(model_proto, out_path)
    print(f"[train_model] Model exported → {out_path}")


def main(csv_path: str) -> None:
    print(f"[train_model] Loading dataset: {csv_path}")
    url_x, dom_x, meta_x, y = load_dataset(csv_path)
    print(f"[train_model] Samples: {len(y)}  Phishing: {int(y.sum())}  Safe: {int((1-y).sum())}")

    model = build_model()
    model.summary()

    label_dict = {
        'output':      y,
        'url_output':  y,
        'dom_output':  y,
        'meta_output': y,
    }

    import tensorflow as tf
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_output_loss',
        mode='min',
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    model.fit(
        [url_x, dom_x, meta_x],
        label_dict,
        epochs=50,
        batch_size=32,
        validation_split=0.15,
        callbacks=[early_stop],
    )

    out_path = os.path.join(os.path.dirname(__file__), 'phishing_model.onnx')
    export_to_onnx(model, out_path)


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'dataset.csv'
    main(csv_path)

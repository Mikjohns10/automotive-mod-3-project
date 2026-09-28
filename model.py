# ============================================================
# DriveGuard AI — CNN + LSTM Model Architecture
# ============================================================
# Hybrid model combining spatial feature extraction (CNN)
# with temporal sequence modeling (LSTM) for driver behaviour
# classification from multimodal sensor streams.
# ============================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
import numpy as np

import config


def build_cnn_lstm_model(
    sequence_length=None,
    num_features=None,
    num_classes=None,
    summary=True,
):
    """
    Build the CNN-LSTM hybrid model.

    Architecture:
    ┌────────────────────────────────────────────────────────┐
    │  Input: (batch, sequence_length, num_features)         │
    │                                                        │
    │  ┌─── CNN Branch (Spatial Features) ────┐              │
    │  │  Conv1D(32, 3) → BatchNorm → ReLU   │              │
    │  │  Conv1D(64, 3) → BatchNorm → ReLU   │              │
    │  │  MaxPool(2) → Dropout(0.25)          │              │
    │  └──────────────────────────────────────┘              │
    │                  ↓                                     │
    │  ┌─── LSTM Branch (Temporal) ───────────┐              │
    │  │  LSTM(128, return_sequences=True)     │              │
    │  │  Dropout(0.3)                         │              │
    │  │  LSTM(64, return_sequences=False)     │              │
    │  │  Dropout(0.3)                         │              │
    │  └──────────────────────────────────────┘              │
    │                  ↓                                     │
    │  Dense(128) → BatchNorm → ReLU → Dropout(0.4)         │
    │  Dense(num_classes, softmax)  → class probabilities    │
    └────────────────────────────────────────────────────────┘
    """
    seq_len   = sequence_length or config.SEQUENCE_LENGTH
    n_feat    = num_features or 13   # 8 raw + 5 engineered
    n_classes = num_classes or config.NUM_CLASSES

    # ── Input Layer ──
    inputs = layers.Input(shape=(seq_len, n_feat), name="sensor_input")

    # ── CNN Branch: Spatial Feature Extraction ──
    x = layers.Conv1D(
        filters=config.CNN_FILTERS_1,
        kernel_size=config.CNN_KERNEL_SIZE,
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4),
        name="conv1d_1",
    )(inputs)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Activation("relu", name="relu_1")(x)

    x = layers.Conv1D(
        filters=config.CNN_FILTERS_2,
        kernel_size=config.CNN_KERNEL_SIZE,
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4),
        name="conv1d_2",
    )(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Activation("relu", name="relu_2")(x)

    x = layers.MaxPooling1D(pool_size=config.CNN_POOL_SIZE, name="maxpool")(x)
    x = layers.Dropout(0.25, name="cnn_dropout")(x)

    # ── LSTM Branch: Temporal Sequence Modeling ──
    x = layers.LSTM(
        config.LSTM_UNITS_1,
        return_sequences=True,
        dropout=config.LSTM_DROPOUT,
        recurrent_dropout=0.1,
        name="lstm_1",
    )(x)

    x = layers.LSTM(
        config.LSTM_UNITS_2,
        return_sequences=False,
        dropout=config.LSTM_DROPOUT,
        recurrent_dropout=0.1,
        name="lstm_2",
    )(x)

    # ── Dense Classification Head ──
    x = layers.Dense(
        config.DENSE_UNITS,
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_1",
    )(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Activation("relu", name="relu_dense")(x)
    x = layers.Dropout(config.DENSE_DROPOUT, name="dense_dropout")(x)

    # ── Output Layer ──
    outputs = layers.Dense(
        n_classes,
        activation="softmax",
        name="output",
    )(x)

    # ── Build Model ──
    model = Model(inputs=inputs, outputs=outputs, name="DriveGuard_CNN_LSTM")

    # ── Compile ──
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    if summary:
        print("\n" + "=" * 60)
        print("  DriveGuard AI — CNN-LSTM Model Architecture")
        print("=" * 60)
        model.summary()

    return model


def get_training_callbacks():
    """Return standard Keras callbacks for training."""
    callbacks = [
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        # Learning rate reduction on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1,
        ),
        # Model checkpoint
        keras.callbacks.ModelCheckpoint(
            filepath=config.MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=config.LOG_DIR,
            histogram_freq=1,
        ),
    ]
    return callbacks


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_cnn_lstm_model(summary=True)
    print(f"\n✓ Model built successfully.")
    print(f"  Total parameters: {model.count_params():,}")

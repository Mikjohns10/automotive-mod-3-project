# ============================================================
# DriveGuard AI — Training Pipeline
# ============================================================
# End-to-end training: data generation → preprocessing →
# model training → evaluation → metrics export.
# ============================================================

import numpy as np
import os
import json
import time
import joblib

import config
from data_generator import generate_dataset
from preprocessing import load_dataset, preprocess_for_training
from model import build_cnn_lstm_model, get_training_callbacks


def train(force_regenerate=False):
    """
    Full training pipeline.

    Steps:
        1. Generate synthetic data (if not exists)
        2. Preprocess & create sequences
        3. Build CNN-LSTM model
        4. Train with callbacks
        5. Evaluate on validation set
        6. Save model, metrics, and history
    """
    print("=" * 60)
    print("  DriveGuard AI — Training Pipeline")
    print("=" * 60)
    t_start = time.time()

    # ── Step 1: Data ──
    csv_path = os.path.join(config.DATA_DIR, "driving_dataset.csv")
    if not os.path.exists(csv_path) or force_regenerate:
        print("\n[1/5] Generating synthetic dataset...")
        generate_dataset()
    else:
        print(f"\n[1/5] Dataset found at {csv_path}")

    # ── Step 2: Preprocess ──
    print("\n[2/5] Preprocessing & feature engineering...")
    df = load_dataset(csv_path)
    X_train, X_val, y_train, y_val, scaler = preprocess_for_training(df)

    # ── Step 3: Build Model ──
    print("\n[3/5] Building CNN-LSTM model...")
    num_features = X_train.shape[2]
    model = build_cnn_lstm_model(
        sequence_length=X_train.shape[1],
        num_features=num_features,
        num_classes=config.NUM_CLASSES,
        summary=True,
    )

    # ── Step 4: Train ──
    print("\n[4/5] Training...")
    print(f"  Epochs     : {config.EPOCHS}")
    print(f"  Batch Size : {config.BATCH_SIZE}")
    print(f"  LR         : {config.LEARNING_RATE}")
    print()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=get_training_callbacks(),
        verbose=1,
    )

    # ── Step 5: Evaluate ──
    print("\n[5/5] Evaluating on validation set...")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"  Validation Loss     : {val_loss:.4f}")
    print(f"  Validation Accuracy : {val_acc:.4f} ({val_acc*100:.2f}%)")

    # Per-class evaluation
    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

    print("\n  Classification Report:")
    print(classification_report(
        y_val, y_pred,
        target_names=config.CLASSES,
        digits=4,
    ))

    cm = confusion_matrix(y_val, y_pred)
    print("  Confusion Matrix:")
    print(cm)

    # ── Save Metrics ──
    metrics = {
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "total_params": int(model.count_params()),
        "sequence_length": int(X_train.shape[1]),
        "num_features": int(num_features),
        "num_classes": config.NUM_CLASSES,
        "classes": config.CLASSES,
        "epochs_completed": len(history.history["loss"]),
        "best_val_accuracy": float(max(history.history.get("val_accuracy", [0]))),
        "confusion_matrix": cm.tolist(),
        "training_time_seconds": round(time.time() - t_start, 2),
    }

    metrics_path = os.path.join(config.MODEL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save training history
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    joblib.dump(hist_dict, config.HISTORY_PATH)

    t_total = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  ✓ Training Complete!")
    print(f"  ✓ Model saved     : {config.MODEL_PATH}")
    print(f"  ✓ Metrics saved   : {metrics_path}")
    print(f"  ✓ Scaler saved    : {config.SCALER_PATH}")
    print(f"  ✓ Time elapsed    : {t_total:.1f}s")
    print(f"  ✓ Best val acc    : {metrics['best_val_accuracy']*100:.2f}%")
    print(f"{'=' * 60}")

    return model, history, metrics


def plot_training_history():
    """Generate and save training plots from saved history."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not os.path.exists(config.HISTORY_PATH):
        print("No training history found.")
        return

    hist = joblib.load(config.HISTORY_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DriveGuard AI — Training History", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(hist["accuracy"], label="Train Acc", linewidth=2)
    axes[0].plot(hist["val_accuracy"], label="Val Acc", linewidth=2)
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(hist["loss"], label="Train Loss", linewidth=2)
    axes[1].plot(hist["val_loss"], label="Val Loss", linewidth=2)
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(config.MODEL_DIR, "training_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"✓ Training plot saved to {plot_path}")

    # Confusion matrix heatmap
    metrics_path = os.path.join(config.MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

        cm = np.array(metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=config.CLASSES,
            yticklabels=config.CLASSES,
            ax=ax,
        )
        ax.set_title("DriveGuard AI — Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        cm_path = os.path.join(config.MODEL_DIR, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"✓ Confusion matrix saved to {cm_path}")


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DriveGuard AI Training")
    parser.add_argument("--regenerate", action="store_true",
                        help="Force regenerate dataset")
    parser.add_argument("--plot", action="store_true",
                        help="Only plot training history (no training)")
    args = parser.parse_args()

    if args.plot:
        plot_training_history()
    else:
        model, history, metrics = train(force_regenerate=args.regenerate)
        plot_training_history()

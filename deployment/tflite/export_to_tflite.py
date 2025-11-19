"""
Export models to TensorFlow Lite format for mobile/edge deployment.

This script creates TensorFlow Lite models with various quantization options:
- FP32 (full precision)
- FP16 (half precision)
- INT8 (integer quantization with calibration)
"""

import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def create_tf_model_from_predictions(X_train, y_train, n_features=52, n_outputs=1):
    """
    Create a simple TF model that mimics the LightGBM predictions.

    Note: Since LightGBM is not directly convertible to TFLite,
    we create a neural network that approximates the LightGBM model.

    Args:
        X_train: Training features
        y_train: Training targets
        n_features: Number of input features
        n_outputs: Number of output predictions

    Returns:
        Trained TensorFlow model
    """
    print(f"\n   Creating TensorFlow approximation model...")
    print(f"   Architecture: {n_features} -> 128 -> 64 -> 32 -> {n_outputs}")

    # Create a neural network to approximate LightGBM
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(n_outputs)
    ])

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    # Train
    print(f"   Training approximation model...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=256,
        validation_split=0.2,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True,
                monitor='val_loss'
            )
        ]
    )

    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    print(f"   Training completed: loss={final_loss:.4f}, val_loss={final_val_loss:.4f}")

    return model


def load_lightgbm_and_create_predictions(model_path, X_data):
    """
    Load LightGBM model and create predictions for training TF model.

    Args:
        model_path: Path to LightGBM pickle file
        X_data: Input features

    Returns:
        Predictions from LightGBM model
    """
    with open(model_path, 'rb') as f:
        lgb_model = pickle.load(f)

    predictions = lgb_model.predict(X_data)
    return predictions


def convert_to_tflite(
    tf_model,
    output_path: str,
    quantization: str = 'fp32',
    representative_dataset=None,
    model_name: str = 'model'
):
    """
    Convert TensorFlow model to TFLite format.

    Args:
        tf_model: TensorFlow/Keras model
        output_path: Path to save TFLite model
        quantization: 'fp32', 'fp16', or 'int8'
        representative_dataset: Generator for INT8 calibration
        model_name: Name for the model

    Returns:
        dict: Conversion statistics
    """
    print(f"\n{'='*70}")
    print(f"CONVERTING TO TFLITE - {quantization.upper()}")
    print(f"{'='*70}")

    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)

    # Apply quantization settings
    if quantization == 'fp16':
        print("   Applying FP16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == 'int8':
        print("   Applying INT8 quantization...")
        if representative_dataset is None:
            raise ValueError("INT8 quantization requires representative_dataset")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        print("   Using FP32 (full precision)...")

    # Convert
    print("   Converting model...")
    tflite_model = converter.convert()

    # Save
    print(f"   Saving to: {output_path}")
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    # Get file size
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"   TFLite model size: {size_mb:.3f} MB")

    # Validate model
    print("\n   Validating TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Input dtype: {input_details[0]['dtype']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    print(f"   Output dtype: {output_details[0]['dtype']}")

    # Test inference
    print("\n   Testing inference...")
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    # Create test input
    if input_dtype == np.int8:
        test_input = np.random.randint(-128, 127, input_shape, dtype=np.int8)
    else:
        test_input = np.random.randn(*input_shape).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    print(f"   Test output: {output.flatten()[:3]}... (showing first 3 values)")
    print(f"   ✓ Inference successful")

    stats = {
        'model_name': model_name,
        'quantization': quantization,
        'size_mb': round(size_mb, 3),
        'input_shape': str(input_details[0]['shape'].tolist()),
        'input_dtype': str(input_details[0]['dtype']),
        'output_shape': str(output_details[0]['shape'].tolist()),
        'output_dtype': str(output_details[0]['dtype']),
    }

    print(f"\n{'='*70}")
    print(f"TFLITE CONVERSION COMPLETED")
    print(f"{'='*70}\n")

    return stats


def export_all_tflite_models():
    """
    Export all LightGBM models to TFLite with multiple quantization options.
    """
    print(f"\n{'#'*70}")
    print(f"# TFLITE EXPORT - SPRINT 6")
    print(f"{'#'*70}\n")

    # Define paths
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / 'models'
    data_dir = project_root / 'data' / 'processed'
    output_dir = Path(__file__).parent

    # Load data for training TF approximation
    print("Loading processed data...")
    try:
        X_train = np.load(data_dir / 'X_train_scaled.npy')
        y_train = np.load(data_dir / 'y_train_scaled.npy')
        X_val = np.load(data_dir / 'X_val_scaled.npy')
        print(f"   X_train shape: {X_train.shape}")
        print(f"   y_train shape: {y_train.shape}")
    except FileNotFoundError:
        print("   Trying CSV format...")
        import pandas as pd
        X_train = pd.read_csv(data_dir / 'X_train.csv').values
        y_train = pd.read_csv(data_dir / 'y_train.csv').values
        X_val = pd.read_csv(data_dir / 'X_val.csv').values
        print(f"   X_train shape: {X_train.shape}")
        print(f"   y_train shape: {y_train.shape}")

    # Load LightGBM model
    lightgbm_model_path = models_dir / 'lightgbm_model.pkl'

    if not lightgbm_model_path.exists():
        print(f"ERROR: LightGBM model not found at {lightgbm_model_path}")
        return

    with open(lightgbm_model_path, 'rb') as f:
        models_dict = pickle.load(f)

    all_stats = []

    # Check if it's a dictionary of models or single model
    if isinstance(models_dict, dict):
        targets = list(models_dict.keys())
        print(f"\nFound {len(targets)} target models: {targets}")
    else:
        targets = ['single_model']
        models_dict = {'single_model': models_dict}

    # Process each target
    for idx, (target_name, lgb_model) in enumerate(models_dict.items()):
        print(f"\n{'='*70}")
        print(f"Processing target: {target_name} ({idx+1}/{len(targets)})")
        print(f"{'='*70}")

        # Get LightGBM predictions to train TF model
        print("\n1. Generating LightGBM predictions for training...")
        if y_train.ndim == 1:
            y_target = y_train
        else:
            y_target = y_train[:, idx]

        lgb_predictions = lgb_model.predict(X_train)

        # Calculate LightGBM performance
        from sklearn.metrics import r2_score, mean_absolute_error
        lgb_r2 = r2_score(y_target, lgb_predictions)
        lgb_mae = mean_absolute_error(y_target, lgb_predictions)
        print(f"   LightGBM performance: R²={lgb_r2:.4f}, MAE={lgb_mae:.4f}")

        # Create TF model to approximate LightGBM
        print("\n2. Creating TensorFlow approximation model...")
        tf_model = create_tf_model_from_predictions(
            X_train,
            lgb_predictions,  # Train TF model to match LightGBM
            n_features=X_train.shape[1],
            n_outputs=1
        )

        # Validate TF model vs LightGBM
        tf_predictions = tf_model.predict(X_train, verbose=0).flatten()
        tf_vs_lgb_r2 = r2_score(lgb_predictions, tf_predictions)
        print(f"   TF approximation quality: R²={tf_vs_lgb_r2:.4f} vs LightGBM")

        # Representative dataset for INT8 quantization
        def representative_dataset_gen():
            for i in range(100):
                # Use validation data for calibration
                start_idx = i * 10
                end_idx = start_idx + 1
                sample = X_val[start_idx:end_idx].astype(np.float32)
                yield [sample]

        # Export to TFLite with different quantization options
        quantization_types = ['fp32', 'fp16', 'int8']

        for quant in quantization_types:
            print(f"\n3. Exporting to TFLite ({quant.upper()})...")

            output_filename = f'lightgbm_{target_name.lower()}_{quant}.tflite'
            output_path = output_dir / output_filename

            rep_dataset = representative_dataset_gen if quant == 'int8' else None

            stats = convert_to_tflite(
                tf_model,
                str(output_path),
                quantization=quant,
                representative_dataset=rep_dataset,
                model_name=f'lightgbm_{target_name}_{quant}'
            )

            stats['target'] = target_name
            stats['lightgbm_r2'] = round(lgb_r2, 4)
            stats['lightgbm_mae'] = round(lgb_mae, 4)
            stats['tf_approximation_r2'] = round(tf_vs_lgb_r2, 4)
            all_stats.append(stats)

    # Save statistics
    stats_path = output_dir / 'tflite_export_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"TFLITE EXPORT SUMMARY")
    print(f"{'='*70}\n")
    print(f"{'Model':<30} {'Quantization':<15} {'Size (MB)':<12} {'TF R²':<10}")
    print(f"{'-'*70}")
    for stat in all_stats:
        model_name = f"{stat['target']}"
        print(f"{model_name:<30} {stat['quantization'].upper():<15} {stat['size_mb']:<12.3f} {stat['tf_approximation_r2']:<10.4f}")

    print(f"\n✓ All TFLite models exported successfully!")
    print(f"✓ Statistics saved to: {stats_path}")

    return all_stats


if __name__ == '__main__':
    try:
        stats = export_all_tflite_models()
        print("\n✓ TFLite export completed successfully!")
        print(f"✓ Models ready for mobile/edge deployment")
    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

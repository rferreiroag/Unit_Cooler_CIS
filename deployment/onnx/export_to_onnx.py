"""
Export LightGBM models to ONNX format for edge deployment.

This script converts trained LightGBM models to ONNX format,
enabling deployment on edge devices with ONNX Runtime.
"""

import pickle
import numpy as np
from pathlib import Path
import json
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import onnxmltools
    from onnxmltools.convert import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType
    import onnxruntime as rt
    import lightgbm as lgb
    ONNX_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Required libraries not installed: {e}")
    print("Install with: pip install onnxmltools onnxruntime lightgbm")
    ONNX_AVAILABLE = False


def export_lightgbm_to_onnx(
    model_path: str,
    output_path: str,
    n_features: int = 52,
    model_name: str = "lightgbm_model"
):
    """
    Export a LightGBM model to ONNX format.

    Args:
        model_path: Path to the pickled LightGBM model
        output_path: Path to save the ONNX model
        n_features: Number of input features (default: 52 from Sprint 1)
        model_name: Name for the ONNX model

    Returns:
        dict: Export statistics (file sizes, validation results)
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX conversion libraries not available")

    print(f"\n{'='*70}")
    print(f"EXPORTING LIGHTGBM MODEL TO ONNX")
    print(f"{'='*70}")

    # Load the trained model
    print(f"\n1. Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    original_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB
    print(f"   Original model size: {original_size:.2f} MB")

    # Define input type for ONNX conversion
    # LightGBM expects float32 input with shape [batch_size, n_features]
    initial_type = [('float_input', FloatTensorType([None, n_features]))]

    # Convert to ONNX
    print(f"\n2. Converting to ONNX format...")
    try:
        # Use onnxmltools for LightGBM conversion
        onnx_model = convert_lightgbm(
            model,
            initial_types=initial_type,
            target_opset=12  # ONNX opset version (compatible with most runtimes)
        )
    except Exception as e:
        print(f"   Error during conversion: {e}")
        raise

    # Save ONNX model
    print(f"\n3. Saving ONNX model to: {output_path}")
    with open(output_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())

    onnx_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
    print(f"   ONNX model size: {onnx_size:.2f} MB")
    print(f"   Size ratio: {onnx_size/original_size:.2f}x")

    # Validate ONNX model
    print(f"\n4. Validating ONNX model...")
    sess = rt.InferenceSession(output_path, providers=['CPUExecutionProvider'])

    # Get input/output info
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    output_shape = sess.get_outputs()[0].shape

    print(f"   Input name: {input_name}")
    print(f"   Input shape: {input_shape}")
    print(f"   Output name: {output_name}")
    print(f"   Output shape: {output_shape}")

    # Test inference with dummy data
    print(f"\n5. Testing inference with dummy data...")
    dummy_input = np.random.randn(1, n_features).astype(np.float32)

    # Original model prediction
    pred_original = model.predict(dummy_input)

    # ONNX model prediction
    pred_onnx = sess.run([output_name], {input_name: dummy_input})[0]

    # Compare predictions
    max_diff = np.max(np.abs(pred_original - pred_onnx.flatten()))
    print(f"   Original prediction: {pred_original}")
    print(f"   ONNX prediction: {pred_onnx.flatten()}")
    print(f"   Max difference: {max_diff:.2e}")

    if max_diff < 1e-5:
        print(f"   ✓ VALIDATION PASSED: Predictions match within tolerance")
        validation_status = "PASSED"
    else:
        print(f"   ⚠ WARNING: Predictions differ by {max_diff:.2e}")
        validation_status = "WARNING"

    # Compile statistics
    stats = {
        'model_name': model_name,
        'original_size_mb': round(original_size, 3),
        'onnx_size_mb': round(onnx_size, 3),
        'size_ratio': round(onnx_size/original_size, 3),
        'n_features': n_features,
        'input_name': input_name,
        'output_name': output_name,
        'input_shape': str(input_shape),
        'output_shape': str(output_shape),
        'max_prediction_diff': float(max_diff),
        'validation_status': validation_status,
        'opset_version': 12
    }

    print(f"\n{'='*70}")
    print(f"EXPORT COMPLETED SUCCESSFULLY")
    print(f"{'='*70}\n")

    return stats


def export_all_models():
    """
    Export all target-specific LightGBM models to ONNX.

    This function handles the three separate models for:
    - UCAOT (Unit Cooler Air Outlet Temperature)
    - UCWOT (Unit Cooler Water Outlet Temperature)
    - UCAF (Unit Cooler Air Flow)
    """
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / 'models'
    output_dir = Path(__file__).parent

    # Check if multi-target model exists (from Sprint 2/5)
    lightgbm_model_path = models_dir / 'lightgbm_model.pkl'

    print(f"\n{'#'*70}")
    print(f"# LIGHTGBM TO ONNX EXPORT - SPRINT 6")
    print(f"{'#'*70}\n")

    all_stats = []

    # The model from Sprint 2 is a single pickle file containing all three target models
    # We need to load it and export each target separately
    if lightgbm_model_path.exists():
        print(f"Found LightGBM model: {lightgbm_model_path}")

        # Load the model to check its structure
        with open(lightgbm_model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Check if it's the new structure with metadata
        if isinstance(model_data, dict) and 'models' in model_data:
            # New structure: {'models': {...}, 'model_type': ..., 'target_names': ...}
            models_dict = model_data['models']
            target_names = model_data.get('target_names', list(models_dict.keys()))
            print(f"Model contains {len(models_dict)} target models: {target_names}")

            for target_name, model in models_dict.items():
                print(f"\nExporting model for target: {target_name}")
                output_path = output_dir / f'lightgbm_{target_name.lower()}.onnx'

                # Save individual model temporarily
                temp_model_path = output_dir / f'temp_{target_name}.pkl'
                with open(temp_model_path, 'wb') as f:
                    pickle.dump(model, f)

                # Export to ONNX
                stats = export_lightgbm_to_onnx(
                    model_path=str(temp_model_path),
                    output_path=str(output_path),
                    n_features=52,
                    model_name=f'lightgbm_{target_name}'
                )
                stats['target'] = target_name
                all_stats.append(stats)

                # Remove temporary file
                temp_model_path.unlink()
        elif isinstance(model_data, dict):
            # Old structure: direct dict of models
            print(f"Model contains {len(model_data)} target models")

            for target_name, model in model_data.items():
                print(f"\nExporting model for target: {target_name}")
                output_path = output_dir / f'lightgbm_{target_name.lower()}.onnx'

                # Save individual model temporarily
                temp_model_path = output_dir / f'temp_{target_name}.pkl'
                with open(temp_model_path, 'wb') as f:
                    pickle.dump(model, f)

                # Export to ONNX
                stats = export_lightgbm_to_onnx(
                    model_path=str(temp_model_path),
                    output_path=str(output_path),
                    n_features=52,
                    model_name=f'lightgbm_{target_name}'
                )
                stats['target'] = target_name
                all_stats.append(stats)

                # Remove temporary file
                temp_model_path.unlink()
        else:
            # Single model - export as-is
            print(f"Exporting single model")
            output_path = output_dir / 'lightgbm_model.onnx'
            stats = export_lightgbm_to_onnx(
                model_path=str(lightgbm_model_path),
                output_path=str(output_path),
                n_features=52,
                model_name='lightgbm_model'
            )
            all_stats.append(stats)
    else:
        print(f"ERROR: LightGBM model not found at {lightgbm_model_path}")
        print(f"Please ensure Sprint 2/5 models are trained first.")
        return

    # Save statistics
    stats_path = output_dir / 'onnx_export_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n{'='*70}")
    print(f"ALL MODELS EXPORTED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"\nExport statistics saved to: {stats_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"EXPORT SUMMARY")
    print(f"{'='*70}\n")
    print(f"{'Model':<25} {'Original (MB)':<15} {'ONNX (MB)':<15} {'Status':<10}")
    print(f"{'-'*70}")
    for stat in all_stats:
        model_name = stat.get('target', stat['model_name'])
        print(f"{model_name:<25} {stat['original_size_mb']:<15.3f} {stat['onnx_size_mb']:<15.3f} {stat['validation_status']:<10}")

    total_original = sum(s['original_size_mb'] for s in all_stats)
    total_onnx = sum(s['onnx_size_mb'] for s in all_stats)
    print(f"{'-'*70}")
    print(f"{'TOTAL':<25} {total_original:<15.3f} {total_onnx:<15.3f}")
    print(f"\nOverall compression: {total_onnx/total_original:.2f}x")

    return all_stats


if __name__ == '__main__':
    if not ONNX_AVAILABLE:
        print("\nERROR: Required libraries not installed")
        print("Install with:")
        print("  pip install skl2onnx onnxruntime")
        sys.exit(1)

    try:
        stats = export_all_models()
        print("\n✓ ONNX export completed successfully!")
        print(f"✓ Models ready for edge deployment")
    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

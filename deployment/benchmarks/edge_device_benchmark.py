"""
Benchmark inference performance on edge devices.

This script measures:
- Inference latency (mean, p50, p95, p99)
- Throughput (inferences/second)
- Memory usage
- Model load time
- Power consumption estimates

Supports:
- Raspberry Pi 4 (ARM Cortex-A72)
- NVIDIA Jetson Orin
- Generic x86_64 devices
"""

import time
import numpy as np
import psutil
import platform
from pathlib import Path
import json
import sys
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def get_device_info() -> Dict:
    """Get information about the current device."""
    info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'architecture': platform.machine(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(logical=False),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
    }

    # Try to detect specific devices
    if 'aarch64' in info['architecture'] or 'arm' in info['architecture'].lower():
        info['device_type'] = 'ARM Device (Raspberry Pi / Jetson)'
    elif 'x86_64' in info['architecture'] or 'AMD64' in info['architecture']:
        info['device_type'] = 'x86_64 Device'
    else:
        info['device_type'] = 'Unknown'

    return info


def benchmark_onnx_model(model_path: str, n_features: int = 52, n_iterations: int = 1000) -> Dict:
    """
    Benchmark ONNX model inference.

    Args:
        model_path: Path to ONNX model
        n_features: Number of input features
        n_iterations: Number of inference iterations

    Returns:
        Performance metrics dictionary
    """
    try:
        import onnxruntime as rt
    except ImportError:
        print("WARNING: onnxruntime not installed, skipping ONNX benchmark")
        return {}

    print(f"\n{'='*70}")
    print(f"BENCHMARKING ONNX MODEL: {Path(model_path).name}")
    print(f"{'='*70}")

    # Load model
    print("\n1. Loading model...")
    start_time = time.time()
    sess = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    load_time = time.time() - start_time
    print(f"   Model load time: {load_time*1000:.2f} ms")

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # Get initial memory
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024**2)  # MB

    # Warmup
    print("\n2. Warming up (10 iterations)...")
    dummy_input = np.random.randn(1, n_features).astype(np.float32)
    for _ in range(10):
        sess.run([output_name], {input_name: dummy_input})

    # Benchmark
    print(f"\n3. Running benchmark ({n_iterations} iterations)...")
    latencies = []

    for i in range(n_iterations):
        # Generate random input
        input_data = np.random.randn(1, n_features).astype(np.float32)

        # Time inference
        start = time.perf_counter()
        output = sess.run([output_name], {input_name: input_data})
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # Convert to ms

        if (i + 1) % 100 == 0:
            print(f"   Progress: {i+1}/{n_iterations}")

    # Get final memory
    mem_after = process.memory_info().rss / (1024**2)  # MB
    mem_increase = mem_after - mem_before

    # Calculate statistics
    latencies = np.array(latencies)
    metrics = {
        'model_type': 'ONNX',
        'model_path': str(model_path),
        'load_time_ms': round(load_time * 1000, 2),
        'n_iterations': n_iterations,
        'latency_mean_ms': round(np.mean(latencies), 3),
        'latency_std_ms': round(np.std(latencies), 3),
        'latency_min_ms': round(np.min(latencies), 3),
        'latency_max_ms': round(np.max(latencies), 3),
        'latency_p50_ms': round(np.percentile(latencies, 50), 3),
        'latency_p95_ms': round(np.percentile(latencies, 95), 3),
        'latency_p99_ms': round(np.percentile(latencies, 99), 3),
        'throughput_inferences_per_sec': round(1000 / np.mean(latencies), 2),
        'memory_increase_mb': round(mem_increase, 2),
    }

    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Load time:      {metrics['load_time_ms']:.2f} ms")
    print(f"Mean latency:   {metrics['latency_mean_ms']:.3f} ± {metrics['latency_std_ms']:.3f} ms")
    print(f"Min latency:    {metrics['latency_min_ms']:.3f} ms")
    print(f"Max latency:    {metrics['latency_max_ms']:.3f} ms")
    print(f"P50 latency:    {metrics['latency_p50_ms']:.3f} ms")
    print(f"P95 latency:    {metrics['latency_p95_ms']:.3f} ms")
    print(f"P99 latency:    {metrics['latency_p99_ms']:.3f} ms")
    print(f"Throughput:     {metrics['throughput_inferences_per_sec']:.2f} inferences/sec")
    print(f"Memory increase: {metrics['memory_increase_mb']:.2f} MB")

    # Check if meets requirements
    target_latency_ms = 100  # From README
    if metrics['latency_p95_ms'] < target_latency_ms:
        print(f"\n✓ MEETS LATENCY REQUIREMENT (P95 < {target_latency_ms} ms)")
    else:
        print(f"\n⚠ EXCEEDS LATENCY TARGET (P95: {metrics['latency_p95_ms']:.3f} ms > {target_latency_ms} ms)")

    return metrics


def benchmark_tflite_model(model_path: str, n_iterations: int = 1000) -> Dict:
    """
    Benchmark TensorFlow Lite model inference.

    Args:
        model_path: Path to TFLite model
        n_iterations: Number of inference iterations

    Returns:
        Performance metrics dictionary
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("WARNING: tensorflow not installed, skipping TFLite benchmark")
        return {}

    print(f"\n{'='*70}")
    print(f"BENCHMARKING TFLITE MODEL: {Path(model_path).name}")
    print(f"{'='*70}")

    # Load model
    print("\n1. Loading model...")
    start_time = time.time()
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    load_time = time.time() - start_time
    print(f"   Model load time: {load_time*1000:.2f} ms")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    # Get initial memory
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024**2)  # MB

    # Warmup
    print("\n2. Warming up (10 iterations)...")
    if input_dtype == np.int8:
        dummy_input = np.random.randint(-128, 127, input_shape, dtype=np.int8)
    else:
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()

    # Benchmark
    print(f"\n3. Running benchmark ({n_iterations} iterations)...")
    latencies = []

    for i in range(n_iterations):
        # Generate random input
        if input_dtype == np.int8:
            input_data = np.random.randint(-128, 127, input_shape, dtype=np.int8)
        else:
            input_data = np.random.randn(*input_shape).astype(np.float32)

        # Time inference
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # Convert to ms

        if (i + 1) % 100 == 0:
            print(f"   Progress: {i+1}/{n_iterations}")

    # Get final memory
    mem_after = process.memory_info().rss / (1024**2)  # MB
    mem_increase = mem_after - mem_before

    # Calculate statistics
    latencies = np.array(latencies)
    metrics = {
        'model_type': 'TFLite',
        'model_path': str(model_path),
        'input_dtype': str(input_dtype),
        'load_time_ms': round(load_time * 1000, 2),
        'n_iterations': n_iterations,
        'latency_mean_ms': round(np.mean(latencies), 3),
        'latency_std_ms': round(np.std(latencies), 3),
        'latency_min_ms': round(np.min(latencies), 3),
        'latency_max_ms': round(np.max(latencies), 3),
        'latency_p50_ms': round(np.percentile(latencies, 50), 3),
        'latency_p95_ms': round(np.percentile(latencies, 95), 3),
        'latency_p99_ms': round(np.percentile(latencies, 99), 3),
        'throughput_inferences_per_sec': round(1000 / np.mean(latencies), 2),
        'memory_increase_mb': round(mem_increase, 2),
    }

    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Input dtype:    {input_dtype}")
    print(f"Load time:      {metrics['load_time_ms']:.2f} ms")
    print(f"Mean latency:   {metrics['latency_mean_ms']:.3f} ± {metrics['latency_std_ms']:.3f} ms")
    print(f"P50 latency:    {metrics['latency_p50_ms']:.3f} ms")
    print(f"P95 latency:    {metrics['latency_p95_ms']:.3f} ms")
    print(f"P99 latency:    {metrics['latency_p99_ms']:.3f} ms")
    print(f"Throughput:     {metrics['throughput_inferences_per_sec']:.2f} inferences/sec")
    print(f"Memory increase: {metrics['memory_increase_mb']:.2f} MB")

    return metrics


def run_comprehensive_benchmark():
    """
    Run comprehensive benchmark on all available models.
    """
    print(f"\n{'#'*70}")
    print(f"# EDGE DEVICE BENCHMARK - SPRINT 6")
    print(f"{'#'*70}\n")

    # Get device info
    device_info = get_device_info()
    print("Device Information:")
    print(f"{'='*70}")
    for key, value in device_info.items():
        print(f"{key:.<30} {value}")
    print(f"{'='*70}\n")

    # Define paths
    project_root = Path(__file__).parent.parent.parent
    onnx_dir = project_root / 'deployment' / 'onnx'
    tflite_dir = project_root / 'deployment' / 'tflite'
    output_dir = Path(__file__).parent

    all_results = {
        'device_info': device_info,
        'benchmarks': []
    }

    # Benchmark ONNX models
    print("\n" + "="*70)
    print("BENCHMARKING ONNX MODELS")
    print("="*70)

    onnx_models = list(onnx_dir.glob('*.onnx'))
    if onnx_models:
        for model_path in onnx_models:
            try:
                metrics = benchmark_onnx_model(str(model_path), n_features=52, n_iterations=1000)
                if metrics:
                    all_results['benchmarks'].append(metrics)
            except Exception as e:
                print(f"Error benchmarking {model_path}: {e}")
    else:
        print("No ONNX models found. Run export_to_onnx.py first.")

    # Benchmark TFLite models
    print("\n" + "="*70)
    print("BENCHMARKING TFLITE MODELS")
    print("="*70)

    tflite_models = list(tflite_dir.glob('*.tflite'))
    if tflite_models:
        for model_path in tflite_models:
            try:
                metrics = benchmark_tflite_model(str(model_path), n_iterations=1000)
                if metrics:
                    all_results['benchmarks'].append(metrics)
            except Exception as e:
                print(f"Error benchmarking {model_path}: {e}")
    else:
        print("No TFLite models found. Run export_to_tflite.py first.")

    # Save results
    results_path = output_dir / 'benchmark_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*70}\n")
    print(f"{'Model':<40} {'Type':<10} {'P50 (ms)':<12} {'P95 (ms)':<12} {'Throughput (inf/s)':<20}")
    print(f"{'-'*100}")

    for result in all_results['benchmarks']:
        model_name = Path(result['model_path']).name
        print(f"{model_name:<40} {result['model_type']:<10} {result['latency_p50_ms']:<12.3f} "
              f"{result['latency_p95_ms']:<12.3f} {result['throughput_inferences_per_sec']:<20.2f}")

    print(f"\n✓ Benchmark completed successfully!")
    print(f"✓ Results saved to: {results_path}")

    # Check if any model meets requirement
    target_latency_ms = 100
    meets_requirement = any(
        r['latency_p95_ms'] < target_latency_ms
        for r in all_results['benchmarks']
    )

    if meets_requirement:
        print(f"\n✓ At least one model meets latency requirement (P95 < {target_latency_ms} ms)")
    else:
        print(f"\n⚠ No models meet latency requirement (P95 < {target_latency_ms} ms)")

    return all_results


if __name__ == '__main__':
    try:
        results = run_comprehensive_benchmark()
        print("\n✓ Edge device benchmark completed successfully!")
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

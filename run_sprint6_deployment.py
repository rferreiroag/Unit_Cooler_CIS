"""
Sprint 6: Edge Deployment - Main Runner Script

This script orchestrates the complete deployment pipeline:
1. Export models to ONNX format
2. Export models to TensorFlow Lite (FP32, FP16, INT8)
3. Run edge device benchmarks
4. Generate deployment report

Usage:
    python run_sprint6_deployment.py [--skip-onnx] [--skip-tflite] [--skip-benchmark]
"""

import argparse
import sys
import subprocess
from pathlib import Path
import time
import json


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def run_step(step_name: str, script_path: str, skip: bool = False):
    """
    Run a deployment step.

    Args:
        step_name: Name of the step for logging
        script_path: Path to the Python script to run
        skip: If True, skip this step

    Returns:
        tuple: (success: bool, elapsed_time: float)
    """
    if skip:
        print(f"⏭️  Skipping {step_name}")
        return True, 0.0

    print_section(step_name)
    print(f"Running: {script_path}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True
        )
        elapsed_time = time.time() - start_time

        print(f"\n✓ {step_name} completed successfully in {elapsed_time:.2f}s")
        return True, elapsed_time

    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ {step_name} failed after {elapsed_time:.2f}s")
        print(f"Error: {e}")
        return False, elapsed_time

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ {step_name} failed with unexpected error: {e}")
        return False, elapsed_time


def check_dependencies():
    """Check if required dependencies are installed."""
    print_section("Checking Dependencies")

    dependencies = {
        'numpy': 'numpy',
        'lightgbm': 'lightgbm',
        'scikit-learn': 'sklearn',
        'onnxruntime (optional)': 'onnxruntime',
        'skl2onnx (optional)': 'skl2onnx',
        'tensorflow (optional)': 'tensorflow',
        'fastapi (optional)': 'fastapi',
    }

    missing = []
    optional_missing = []

    for name, module in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            if 'optional' in name:
                print(f"⚠ {name} (not installed - some features unavailable)")
                optional_missing.append(name)
            else:
                print(f"✗ {name} (REQUIRED)")
                missing.append(name)

    if missing:
        print(f"\n✗ Missing required dependencies: {', '.join(missing)}")
        print(f"Install with: pip install -r requirements.txt")
        return False

    if optional_missing:
        print(f"\n⚠ Optional dependencies not installed:")
        for dep in optional_missing:
            print(f"   - {dep}")
        print(f"\nTo enable all features, install:")
        print(f"   pip install skl2onnx onnxruntime tensorflow fastapi uvicorn")

    return True


def generate_deployment_report(results: dict):
    """Generate a comprehensive deployment report."""
    print_section("Generating Deployment Report")

    project_root = Path(__file__).parent
    deployment_dir = project_root / 'deployment'

    # Collect statistics from each step
    report = {
        'sprint': 'Sprint 6: Edge Deployment',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'execution_times': results['execution_times'],
        'total_time_seconds': results['total_time'],
        'steps_completed': results['steps_completed'],
        'steps_failed': results['steps_failed'],
        'models_exported': {},
        'benchmarks': {}
    }

    # Load ONNX export stats
    onnx_stats_path = deployment_dir / 'onnx' / 'onnx_export_stats.json'
    if onnx_stats_path.exists():
        with open(onnx_stats_path, 'r') as f:
            report['models_exported']['onnx'] = json.load(f)

    # Load TFLite export stats
    tflite_stats_path = deployment_dir / 'tflite' / 'tflite_export_stats.json'
    if tflite_stats_path.exists():
        with open(tflite_stats_path, 'r') as f:
            report['models_exported']['tflite'] = json.load(f)

    # Load benchmark results
    benchmark_path = deployment_dir / 'benchmarks' / 'benchmark_results.json'
    if benchmark_path.exists():
        with open(benchmark_path, 'r') as f:
            report['benchmarks'] = json.load(f)

    # Save report
    report_path = deployment_dir / 'sprint6_deployment_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✓ Deployment report saved to: {report_path}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"  DEPLOYMENT SUMMARY")
    print(f"{'='*80}\n")
    print(f"Total execution time: {results['total_time']:.2f}s")
    print(f"Steps completed: {results['steps_completed']}")
    print(f"Steps failed: {results['steps_failed']}")

    if report['models_exported'].get('onnx'):
        print(f"\n✓ ONNX models exported: {len(report['models_exported']['onnx'])}")

    if report['models_exported'].get('tflite'):
        print(f"✓ TFLite models exported: {len(report['models_exported']['tflite'])}")

    if report['benchmarks'].get('benchmarks'):
        print(f"✓ Benchmarks completed: {len(report['benchmarks']['benchmarks'])}")

    print(f"\n{'='*80}\n")

    return report


def main():
    """Main deployment pipeline."""
    parser = argparse.ArgumentParser(
        description='Sprint 6: Edge Deployment Pipeline'
    )
    parser.add_argument('--skip-onnx', action='store_true',
                        help='Skip ONNX export')
    parser.add_argument('--skip-tflite', action='store_true',
                        help='Skip TensorFlow Lite export')
    parser.add_argument('--skip-benchmark', action='store_true',
                        help='Skip benchmarking')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: skip optional steps')

    args = parser.parse_args()

    print(f"\n{'#'*80}")
    print(f"# SPRINT 6: EDGE DEPLOYMENT PIPELINE")
    print(f"# HVAC Unit Cooler Digital Twin")
    print(f"{'#'*80}\n")

    # Check dependencies
    if not check_dependencies():
        print("\n✗ Dependency check failed. Please install required packages.")
        return 1

    project_root = Path(__file__).parent
    deployment_dir = project_root / 'deployment'

    # Define steps
    steps = [
        {
            'name': 'ONNX Export',
            'script': deployment_dir / 'onnx' / 'export_to_onnx.py',
            'skip': args.skip_onnx or args.quick
        },
        {
            'name': 'TensorFlow Lite Export',
            'script': deployment_dir / 'tflite' / 'export_to_tflite.py',
            'skip': args.skip_tflite
        },
        {
            'name': 'Edge Device Benchmark',
            'script': deployment_dir / 'benchmarks' / 'edge_device_benchmark.py',
            'skip': args.skip_benchmark
        },
    ]

    # Execute pipeline
    results = {
        'execution_times': {},
        'steps_completed': 0,
        'steps_failed': 0,
        'total_time': 0
    }

    pipeline_start = time.time()

    for step in steps:
        if not step['script'].exists():
            print(f"⚠ Warning: Script not found: {step['script']}")
            continue

        success, elapsed = run_step(step['name'], str(step['script']), step['skip'])

        results['execution_times'][step['name']] = elapsed

        if success:
            results['steps_completed'] += 1
        else:
            results['steps_failed'] += 1
            print(f"\n⚠ Warning: {step['name']} failed. Continuing...")

    results['total_time'] = time.time() - pipeline_start

    # Generate report
    report = generate_deployment_report(results)

    # Final status
    if results['steps_failed'] == 0:
        print("✓ Sprint 6 deployment completed successfully!")
        print("\n📦 Deployment artifacts:")
        print(f"   - ONNX models: deployment/onnx/")
        print(f"   - TFLite models: deployment/tflite/")
        print(f"   - Benchmarks: deployment/benchmarks/")
        print(f"   - Docker: deployment/docker/")
        print(f"   - API: api/main.py")
        print("\n🚀 Next steps:")
        print(f"   1. Test API: cd api && uvicorn main:app --reload")
        print(f"   2. Build Docker: cd deployment/docker && docker-compose up")
        print(f"   3. Deploy to edge device")
        return 0
    else:
        print(f"⚠ Sprint 6 completed with {results['steps_failed']} failed steps")
        return 1


if __name__ == '__main__':
    sys.exit(main())

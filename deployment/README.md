# Sprint 6: Edge Deployment Documentation

## 🚀 Overview

This directory contains all deployment artifacts for the HVAC Unit Cooler Digital Twin, optimized for edge devices including Raspberry Pi 4 and NVIDIA Jetson Orin.

## 📂 Directory Structure

```
deployment/
├── onnx/                          # ONNX models and export scripts
│   ├── export_to_onnx.py         # Convert LightGBM to ONNX
│   ├── *.onnx                     # Exported ONNX models
│   └── onnx_export_stats.json    # Export statistics
├── tflite/                        # TensorFlow Lite models
│   ├── export_to_tflite.py       # Convert to TFLite (FP32/FP16/INT8)
│   ├── *_fp32.tflite             # Full precision models
│   ├── *_fp16.tflite             # Half precision models
│   ├── *_int8.tflite             # Quantized INT8 models
│   └── tflite_export_stats.json  # Export statistics
├── quantized/                     # Quantized model artifacts
├── benchmarks/                    # Performance benchmarks
│   ├── edge_device_benchmark.py  # Benchmark script
│   └── benchmark_results.json    # Benchmark results
├── docker/                        # Docker containerization
│   ├── Dockerfile                # Standard Docker image
│   ├── Dockerfile.edge           # Lightweight edge image
│   └── docker-compose.yml        # Docker Compose configuration
└── README.md                      # This file
```

## 🎯 Quick Start

### 1. Run Complete Deployment Pipeline

```bash
# From project root
python run_sprint6_deployment.py
```

This will:
- ✓ Export models to ONNX
- ✓ Export models to TensorFlow Lite (FP32/FP16/INT8)
- ✓ Run edge device benchmarks
- ✓ Generate deployment report

### 2. Export Models Only

```bash
# Export to ONNX
python deployment/onnx/export_to_onnx.py

# Export to TensorFlow Lite
python deployment/tflite/export_to_tflite.py
```

### 3. Run Benchmarks

```bash
python deployment/benchmarks/edge_device_benchmark.py
```

## 🐳 Docker Deployment

### Standard Deployment (x86_64/ARM64)

```bash
cd deployment/docker

# Build and run
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Edge Device Deployment (Raspberry Pi / Jetson)

```bash
cd deployment/docker

# Build lightweight image
docker build -f Dockerfile.edge -t hvac-digital-twin:edge ../..

# Run on edge device
docker run -d \
  --name hvac-twin \
  -p 8000:8000 \
  -v $(pwd)/../../models:/app/models:ro \
  -v $(pwd)/../../data/processed:/app/data/processed:ro \
  --restart unless-stopped \
  hvac-digital-twin:edge
```

## 🔌 API Usage

### Start API Server

```bash
# Local development
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### API Endpoints

**Health Check**
```bash
curl http://localhost:8000/health
```

**Single Prediction**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "UCWIT": 7.5,
    "UCAIT": 25.0,
    "UCWF": 15.0,
    "UCAIH": 50.0,
    "AMBT": 22.0,
    "UCTSP": 21.0
  }'
```

**Batch Prediction**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"UCWIT": 7.5, "UCAIT": 25.0, "UCWF": 15.0, "UCAIH": 50.0, "AMBT": 22.0, "UCTSP": 21.0},
    {"UCWIT": 8.0, "UCAIT": 26.0, "UCWF": 16.0, "UCAIH": 55.0, "AMBT": 23.0, "UCTSP": 22.0}
  ]'
```

**API Documentation**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📊 Model Formats

### ONNX Models
- **Format**: ONNX Runtime compatible
- **Precision**: FP32
- **Size**: ~2-3 MB per model
- **Use Case**: Cross-platform deployment, GPU acceleration
- **Devices**: x86_64, ARM64, NVIDIA GPUs

### TensorFlow Lite Models

**FP32 (Full Precision)**
- Precision: 32-bit floating point
- Size: ~2-4 MB
- Speed: Baseline
- Accuracy: Best

**FP16 (Half Precision)**
- Precision: 16-bit floating point
- Size: ~1-2 MB (50% reduction)
- Speed: 1.5-2× faster
- Accuracy: Minimal loss (<0.1%)

**INT8 (Quantized)**
- Precision: 8-bit integer
- Size: ~0.5-1 MB (75% reduction)
- Speed: 2-4× faster
- Accuracy: Small loss (<1%)

## 🎯 Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Inference Latency (P95) | <100 ms | ⏳ TBD |
| Memory Usage | <2 GB | ⏳ TBD |
| Model Size | <100 MB | ✅ <5 MB |
| Accuracy (R²) | >0.95 | ✅ 0.993-1.0 |

## 🖥️ Edge Device Requirements

### Minimum Requirements

**Raspberry Pi 4**
- Model: 4GB RAM or higher
- OS: Raspberry Pi OS (64-bit)
- Python: 3.10+
- Storage: 2GB free space

**NVIDIA Jetson Orin**
- Model: Nano or higher
- JetPack: 5.0+
- Python: 3.10+
- Storage: 4GB free space

### Installation on Edge Device

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python and dependencies
sudo apt-get install -y python3-pip python3-dev

# Install minimal requirements
pip install -r requirements.edge.txt

# Copy models to device
scp -r models/ data/processed/ user@device:/home/user/hvac-twin/

# Run API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## 📈 Benchmarking

### Run Benchmarks

```bash
python deployment/benchmarks/edge_device_benchmark.py
```

This will measure:
- ✓ Model load time
- ✓ Inference latency (mean, P50, P95, P99)
- ✓ Throughput (inferences/second)
- ✓ Memory usage
- ✓ CPU utilization

### Benchmark Results

Results are saved to:
- `deployment/benchmarks/benchmark_results.json`

Expected performance (on Raspberry Pi 4):
- Load time: <500 ms
- Inference latency (P95): 10-50 ms
- Throughput: 20-100 inf/s
- Memory: <500 MB

## 🔧 Troubleshooting

### Issue: ONNX export fails

**Solution**: Install required packages
```bash
pip install skl2onnx onnxruntime
```

### Issue: TFLite export fails

**Solution**: Install TensorFlow
```bash
pip install tensorflow>=2.13.0
```

### Issue: API can't find models

**Solution**: Check model paths
```bash
ls models/lightgbm_model.pkl
ls data/processed/X_scaler_mean.npy
```

### Issue: Docker build fails on ARM

**Solution**: Use edge Dockerfile
```bash
docker build -f deployment/docker/Dockerfile.edge -t hvac-twin:edge .
```

## 🚀 Production Deployment Checklist

- [ ] Models exported to ONNX/TFLite
- [ ] Benchmarks run and validated
- [ ] API tested with sample data
- [ ] Docker image built and tested
- [ ] Edge device provisioned
- [ ] Network connectivity verified
- [ ] Monitoring/logging configured
- [ ] Health checks enabled
- [ ] Backup/recovery plan
- [ ] Documentation reviewed

## 📚 Additional Resources

- **Main README**: `../README.md`
- **Sprint 5 Evaluation**: `../docs/Sprint5_Comprehensive_Evaluation_Report.md`
- **Sprint 3 PINN Analysis**: `../docs/Sprint3_PINN_Comprehensive_Analysis.md`
- **Data Quality Report**: `../data_quality_report.md`

## 🤝 Support

For issues or questions:
1. Check this documentation
2. Review benchmark results
3. Check API logs: `docker-compose logs -f`
4. Review deployment report: `deployment/sprint6_deployment_report.json`

---

**Last Updated**: 2025-11-18
**Sprint**: 6 (Deployment)
**Status**: ⏳ In Progress

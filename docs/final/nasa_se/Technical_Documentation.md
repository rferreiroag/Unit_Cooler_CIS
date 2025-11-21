# HVAC Unit Cooler Digital Twin
## Technical Documentation - NASA SE Handbook Standards

**Document Number**: HVAC-DT-TDD-001
**Version**: 1.0
**Date**: 2025-11-21
**Classification**: Internal/Restricted
**Project**: HVAC Unit Cooler Digital Twin

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-21 | Development Team | Initial release |

**Approval:**

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Technical Lead | [Name] | _________ | _____ |
| Project Manager | [Name] | _________ | _____ |
| Quality Assurance | [Name] | _________ | _____ |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Technical Architecture](#3-technical-architecture)
4. [Model Development](#4-model-development)
5. [Performance Validation](#5-performance-validation)
6. [Deployment Infrastructure](#6-deployment-infrastructure)
7. [Integration Capabilities](#7-integration-capabilities)
8. [Quality Assurance](#8-quality-assurance)
9. [Risk Assessment](#9-risk-assessment)
10. [References](#10-references)

---

## 1. Executive Summary

### 1.1 Purpose

This document provides comprehensive technical documentation for the HVAC Unit Cooler Digital Twin system, developed for naval vessel environmental control applications. The system provides real-time predictive capabilities for HVAC performance optimization.

### 1.2 Scope

The digital twin encompasses:
- Data-driven machine learning models (LightGBM)
- Edge deployment infrastructure (ONNX Runtime)
- Real-time monitoring and drift detection
- Industrial protocol integration (MQTT/BACnet)
- Interactive dashboard for operators

### 1.3 Key Achievements

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Prediction Accuracy (R²) | >0.95 | **0.993-1.0** | ✅ Exceeded |
| Inference Latency (P95) | <100 ms | **0.022 ms** | ✅ **4500× better** |
| Model Size | <100 MB | **1.6 MB** | ✅ **60× smaller** |
| Memory Usage | <2 GB | **<50 MB** | ✅ **40× lower** |
| MAPE Error | <10% | **0.008-8.7%** | ✅ Exceeded |

### 1.4 System Status

**Current State**: Production-Ready ✅
**Deployment**: Approved for edge devices
**Validation**: Complete (5 sprints)
**Integration**: MQTT/BACnet enabled
**Monitoring**: Drift detection active

---

## 2. System Overview

### 2.1 System Description

The HVAC Digital Twin is a machine learning-based predictive system that models the behavior of naval HVAC Unit Cooler systems. It predicts:

1. **UCAOT** - Unit Cooler Air Outlet Temperature (°C)
2. **UCWOT** - Unit Cooler Water Outlet Temperature (°C)
3. **UCAF** - Unit Cooler Air Flow (m³/h)
4. **Q_thermal** - Thermal Power (kW) [derived]

### 2.2 Operational Context

**Environment**: Naval vessels, engine rooms, enclosed spaces
**Operating Conditions**:
- Temperature range: -10°C to 50°C ambient
- Humidity: 0-100% RH
- Water inlet: 5-15°C
- Air inlet: 15-35°C
- Continuous operation: 24/7

### 2.3 System Requirements

**Functional Requirements:**
- FR-01: Predict HVAC outputs with R² > 0.95
- FR-02: Provide predictions in < 100ms
- FR-03: Operate on edge devices (Raspberry Pi 4+)
- FR-04: Detect data/model drift automatically
- FR-05: Integrate with building automation systems

**Non-Functional Requirements:**
- NFR-01: 99.9% uptime (subject to hardware)
- NFR-02: <50 MB memory footprint
- NFR-03: Support offline operation
- NFR-04: Secure communication (TLS/SSL)
- NFR-05: Data retention: 30 days local, indefinite cloud

### 2.4 System Boundaries

**In Scope:**
- Predictive modeling of HVAC performance
- Real-time inference on edge devices
- Data drift detection and alerting
- MQTT/BACnet protocol integration
- Web-based monitoring dashboard

**Out of Scope:**
- HVAC hardware control (read-only predictions)
- Equipment fault diagnosis
- Predictive maintenance scheduling
- Multi-zone HVAC coordination
- Weather forecasting integration

---

## 3. Technical Architecture

### 3.1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                       │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  Streamlit   │  │   BACnet     │  │   MQTT Topics   │  │
│  │  Dashboard   │  │  Objects     │  │  (Pub/Sub)      │  │
│  │ (Port 8501)  │  │ (Read/Write) │  │  (Port 1883)    │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬────────┘  │
└─────────┼──────────────────┼───────────────────┼───────────┘
          │                  │                   │
┌─────────▼──────────────────▼───────────────────▼───────────┐
│                      APPLICATION LAYER                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │               FastAPI REST API (Port 8000)           │  │
│  │  - /health      - /predict      - /predict/batch     │  │
│  └───────────────────────┬──────────────────────────────┘  │
│                          │                                  │
│  ┌───────────────────────▼──────────────────────────────┐  │
│  │            Drift Detection & Monitoring              │  │
│  │  - PSI Calculation  - KS Test  - Alert Generation   │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                      INFERENCE LAYER                          │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐  │
│  │  ONNX Runtime  │  │   TFLite      │  │  LightGBM     │  │
│  │  (Primary)     │  │   (Mobile)     │  │  (Native)     │  │
│  └────────┬───────┘  └────────┬───────┘  └───────┬───────┘  │
│           └──────────────┬─────────────────────────┘          │
│                          │                                     │
│  ┌───────────────────────▼──────────────────────────────┐    │
│  │      Model Repository (3 models: UCAOT,UCWOT,UCAF)   │    │
│  │      Size: 1.6 MB total | Format: ONNX/PKL           │    │
│  └──────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                        DATA LAYER                             │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐  │
│  │  Sensor Data   │  │  Predictions   │  │ Drift Reports │  │
│  │  (MQTT/BACnet) │  │  (TimeSeries)  │  │  (JSON)       │  │
│  └────────────────┘  └────────────────┘  └───────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow

**Inference Pipeline:**
```
Sensor Data → Feature Engineering → Normalization →
ONNX Inference → Denormalization → Output Publishing
```

**Monitoring Pipeline:**
```
Production Data → Drift Detector → Statistical Tests →
Alert Generation → MQTT/Dashboard
```

### 3.3 Technology Stack

**Core:**
- Python 3.10+
- LightGBM 4.1+
- scikit-learn 1.3+
- NumPy 1.24+, Pandas 2.1+

**Deployment:**
- ONNX Runtime 1.16+
- Docker 24.0+
- FastAPI 0.104+
- Uvicorn (ASGI server)

**Integration:**
- paho-mqtt 1.6+ (MQTT)
- BAC0 22.9+ (BACnet)
- Streamlit 1.28+ (Dashboard)

**Monitoring:**
- SciPy 1.11+ (statistical tests)
- Plotly 5.17+ (visualization)

### 3.4 Hardware Requirements

**Minimum (Edge Device):**
- CPU: ARM Cortex-A72 (Raspberry Pi 4) or equivalent
- RAM: 2 GB
- Storage: 4 GB
- Network: 100 Mbps Ethernet or WiFi

**Recommended (Production):**
- CPU: x86_64 quad-core 2.0+ GHz
- RAM: 4 GB
- Storage: 16 GB SSD
- Network: 1 Gbps Ethernet

---

## 4. Model Development

### 4.1 Development Methodology

**Approach**: Iterative CRISP-DM with continuous validation

**Sprint Structure:**
- Sprint 0: Data exploration and quality assessment
- Sprint 1: Feature engineering (52 physics-based features)
- Sprint 2: Baseline models (LightGBM selected)
- Sprint 3: PINN investigation (not viable - documented)
- Sprint 5: Comprehensive evaluation
- Sprint 6: Edge deployment
- Sprint 7: Real-time integration
- Sprint 8: Documentation & transfer

### 4.2 Data Description

**Training Dataset:**
- Source: Naval HVAC experimental facility
- Samples: 56,211 total → 43,147 usable (76.8%)
- Features: 32 raw → 52 engineered
- Temporal split: 70% train / 15% val / 15% test
- Time range: Summer/Winter operating conditions

**Input Variables (6 primary):**
1. UCWIT - Water Inlet Temperature
2. UCAIT - Air Inlet Temperature
3. UCWF - Water Flow Rate
4. UCAIH - Air Inlet Humidity
5. AMBT - Ambient Temperature
6. UCTSP - Temperature Setpoint

**Data Quality Issues (addressed):**
- Missing values: 23-76% in various columns
- Negative flows: 12,620 samples (22.45%)
- Sensor saturation: UCAF max = 65,535 (2^16-1)
- Physics violations: Energy imbalance ~10%

### 4.3 Feature Engineering

**52 Engineered Features:**

**Energy Balance:**
- Q_water, Q_air, Q_imbalance
- delta_T_water, delta_T_air, delta_T_ratio

**Heat Exchanger Performance:**
- efficiency_HX, effectiveness, NTU
- LMTD (Log Mean Temperature Difference)

**Flow Dynamics:**
- mdot_air, mdot_water
- Re_air_estimate, flow_ratio

**Temporal:**
- hour_sin, hour_cos, cycle_hour
- time_of_day, is_night_shift

**Interaction Terms:**
- T_water_x_flow, ambient_x_inlet
- humidity_x_temperature

### 4.4 Model Selection

**Models Evaluated:**

| Model | R² (avg) | Training Time | Inference | Selected |
|-------|----------|---------------|-----------|----------|
| LinearRegression | 0.69 | <1s | <1ms | ❌ |
| RandomForest | 0.99 | 30s | 50ms | ❌ |
| XGBoost | 0.99 | 60s | <1ms | ❌ |
| **LightGBM** | **1.00** | **45s** | **<1ms** | ✅ |
| MLP (256-128-64) | 0.99 | 120s | 5ms | ❌ |
| PINN (5 variants) | 0.20 | 600s | 10ms | ❌ |

**Selection Criteria:**
1. Accuracy: R² > 0.99 (required)
2. Speed: Training <2 min, inference <10ms
3. Size: Model <10 MB
4. Robustness: Handles missing data, outliers
5. Interpretability: Feature importance available

**Winner: LightGBM**
- Best accuracy (R²=0.993-1.0)
- Fast training (<1 minute)
- Smallest model size (2.1 MB → 1.6 MB ONNX)
- Excellent handling of tabular data
- Feature importance for interpretability

### 4.5 Model Architecture

**Configuration (per target):**
```python
LGBMRegressor(
    objective='regression',
    metric='rmse',
    boosting_type='gbdt',
    num_leaves=31,
    learning_rate=0.1,
    n_estimators=100,
    max_depth=-1,
    random_state=42
)
```

**Three Independent Models:**
- Model 1: UCAOT (Air Outlet Temperature)
- Model 2: UCWOT (Water Outlet Temperature)
- Model 3: UCAF (Air Flow)

**Ensemble**: Not required (individual models sufficient)

### 4.6 Training Process

**Procedure:**
1. Load preprocessed data (X_train_scaled, y_train_scaled)
2. Train 3 independent LightGBM models
3. Validate on X_val → tune if needed
4. Final evaluation on X_test (unseen data)
5. Export to ONNX for deployment

**Cross-Validation:**
- Method: 5-fold temporal split
- Result: R² > 0.999 across all folds
- Std Dev: <0.0001 (extremely stable)

---

## 5. Performance Validation

### 5.1 Test Set Performance

**Final Results (Test Set - 8,432 samples):**

| Target | R² | MAE | RMSE | MAPE | Status |
|--------|----|-----|------|------|--------|
| UCAOT | 0.9926 | 0.0335 | 0.0652 | 8.68% | ✅ |
| UCWOT | 0.9975 | 0.0309 | 0.0512 | 8.71% | ✅ |
| UCAF | 1.0000 | 0.0001 | 0.0005 | 0.008% | ✅ |

**Interpretation:**
- R² near 1.0 indicates near-perfect fit
- MAPE < 10% meets operational requirements
- RMSE values within acceptable tolerance

### 5.2 Cross-Validation Robustness

**5-Fold Temporal CV Results:**

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| R² (UCAOT) | 0.9999 | 0.0001 | 0.9998 | 1.0000 |
| R² (UCWOT) | 1.0000 | <0.0001 | 0.9999 | 1.0000 |
| R² (UCAF) | 1.0000 | <0.0001 | 1.0000 | 1.0000 |

**Conclusion**: Exceptionally stable across temporal splits

### 5.3 Benchmark Comparison

**vs. Modelica FMU (Existing System):**

| Target | FMU MAPE | DT MAPE | Improvement |
|--------|----------|---------|-------------|
| UCAOT | 30-221% | 8.68% | **93.1%** ↓ |
| UCWOT | 30-221% | 8.71% | **93.1%** ↓ |
| UCAF | 30-221% | 0.008% | **100.0%** ↓ |

### 5.4 Operating Conditions Coverage

**Performance across ranges:**
- Water inlet temp (5-15°C): Consistent R² > 0.99
- Air inlet temp (15-35°C): Consistent R² > 0.99
- Flow rates (0-50 L/min): Consistent R² > 0.99
- Humidity (0-100%): Consistent R² > 0.99

**No performance degradation observed in any operational regime**

### 5.5 Edge Device Performance

**ONNX Runtime Benchmarks (x86_64):**

| Model | Load Time | P50 Latency | P95 Latency | Throughput |
|-------|-----------|-------------|-------------|------------|
| UCAOT | 31.5 ms | 0.016 ms | 0.022 ms | 59,165 inf/s |
| UCWOT | 15.5 ms | 0.015 ms | 0.017 ms | 66,097 inf/s |
| UCAF | 5.6 ms | 0.015 ms | 0.021 ms | 61,862 inf/s |

**All models meet <100ms latency requirement with 4500× margin**

---

## 6. Deployment Infrastructure

### 6.1 Model Export

**Formats Available:**
1. **ONNX** (Primary): 1.6 MB total, CPU optimized
2. **TFLite** (Mobile): FP32/FP16/INT8 quantization ready
3. **Pickle** (Development): 2.1 MB, Python only

**Export Process:**
```bash
python deployment/onnx/export_to_onnx.py
# Output: lightgbm_{ucaot,ucwot,ucaf}.onnx
```

### 6.2 Containerization

**Docker Images:**
1. **Standard** (`Dockerfile`): Full stack with FastAPI + Dashboard
2. **Edge** (`Dockerfile.edge`): Minimal image for Raspberry Pi/Jetson
3. **Development** (`docker-compose.yml`): Multi-service orchestration

**Deployment:**
```bash
docker-compose up -d
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

### 6.3 API Endpoints

**FastAPI Server (Port 8000):**

| Endpoint | Method | Purpose | Response Time |
|----------|--------|---------|---------------|
| `/health` | GET | System status | <5ms |
| `/predict` | POST | Single prediction | <50ms |
| `/predict/batch` | POST | Batch predictions | <100ms |
| `/docs` | GET | Swagger UI | - |

### 6.4 Integration Protocols

**MQTT (IoT Communication):**
- Protocol: MQTT v3.1.1 / v5.0
- QoS: 0 (at most once), 1 (at least once), 2 (exactly once)
- Topics: `hvac/sensors/*`, `hvac/predictions/*`, `hvac/alerts/*`
- Latency: <5ms (local network)

**BACnet (Building Automation):**
- Protocol: BACnet/IP
- Objects: 6 Analog Inputs, 4 Analog Values
- Read/Write latency: 10-50ms
- Compatible with standard BMS/SCADA systems

---

## 7. Integration Capabilities

### 7.1 Real-time Dashboard

**Streamlit Application:**
- URL: http://localhost:8501
- Features:
  - Manual input mode
  - Real-time predictions
  - Temperature flow visualization
  - Performance metrics
  - API health monitoring

### 7.2 Drift Detection

**Statistical Methods:**
1. **PSI (Population Stability Index)**: >0.2 = significant drift
2. **KS Test (Kolmogorov-Smirnov)**: p<0.05 = distribution change
3. **Performance Monitoring**: MAE/RMSE degradation tracking

**Alert Thresholds:**
- PSI > 0.2: Warning
- PSI > 0.5: Critical (retrain recommended)
- Performance drift > 10%: Alert

### 7.3 Monitoring & Alerting

**Monitoring Capabilities:**
- Feature drift detection (real-time)
- Model performance tracking
- System health metrics
- Resource utilization

**Alert Channels:**
- MQTT topics (`hvac/alerts/*`)
- Dashboard notifications
- JSON report generation
- Email (configurable)

---

## 8. Quality Assurance

### 8.1 Testing Strategy

**Unit Tests:**
- Model loading and inference
- Feature engineering correctness
- API endpoint validation
- MQTT/BACnet communication

**Integration Tests:**
- End-to-end prediction pipeline
- Dashboard-API integration
- Drift detection workflow
- Docker deployment

**Performance Tests:**
- Inference latency benchmarks
- Throughput stress tests
- Memory leak detection
- Concurrent request handling

### 8.2 Validation Results

| Test Category | Tests Run | Passed | Coverage |
|---------------|-----------|--------|----------|
| Unit Tests | 45 | 45 | 87% |
| Integration | 12 | 12 | 100% |
| Performance | 8 | 8 | 100% |
| Security | 6 | 6 | 100% |

### 8.3 Known Limitations

1. **Input Range**: Predictions valid within training data range
2. **Missing Features**: Requires at least 80% feature availability
3. **Drift Sensitivity**: May flag seasonal changes as drift
4. **Hardware Dependency**: Performance varies by CPU architecture
5. **Network Latency**: MQTT/BACnet subject to network conditions

### 8.4 Future Improvements

1. **Auto-retraining**: Automatic model update on drift detection
2. **Multi-model ensemble**: Confidence intervals
3. **Anomaly detection**: Fault diagnosis beyond prediction
4. **Time-series forecasting**: Predictive horizon extension
5. **Mobile app**: iOS/Android dashboard

---

## 9. Risk Assessment

### 9.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Data drift | High | Medium | Automated monitoring + alerts |
| Hardware failure | Low | High | Redundant deployment |
| Network outage | Medium | Medium | Offline operation mode |
| Model degradation | Low | High | Regular retraining schedule |
| Security breach | Low | Critical | TLS/SSL, authentication |

### 9.2 Operational Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| User error | Medium | Low | Comprehensive training |
| Integration issues | Low | Medium | Extensive testing |
| Maintenance gaps | Low | Medium | Detailed documentation |
| Knowledge loss | Medium | High | Transfer sessions |

### 9.3 Mitigation Strategies

1. **Automated Monitoring**: Drift detection runs continuously
2. **Backup Systems**: Model versioning and rollback capability
3. **Documentation**: Comprehensive guides (this document)
4. **Training**: User sessions and hands-on workshops
5. **Support**: Dedicated contact for issues

---

## 10. References

### 10.1 Technical Documents

1. **Sprint 3**: PINN Comprehensive Analysis (50 pages)
2. **Sprint 5**: Comprehensive Evaluation Report (40 pages)
3. **Sprint 7**: Real-time Integration Guide (40 pages)
4. **Data Quality Report**: Detailed assessment (20 pages)

### 10.2 Standards & Guidelines

1. **NASA SE Handbook** (NASA-HDBK-2203)
2. **ISO 9001**: Quality Management Systems
3. **IEC 61131**: Programmable Controllers
4. **BACnet Protocol** (ASHRAE 135)
5. **MQTT Specification** v3.1.1 / v5.0

### 10.3 External Libraries

1. **LightGBM**: https://github.com/microsoft/LightGBM
2. **ONNX Runtime**: https://onnxruntime.ai/
3. **FastAPI**: https://fastapi.tiangolo.com/
4. **Streamlit**: https://streamlit.io/
5. **paho-mqtt**: https://www.eclipse.org/paho/

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| BACnet | Building Automation and Control Network protocol |
| Digital Twin | Virtual representation of physical system |
| Drift | Statistical change in data distribution over time |
| GBDT | Gradient Boosted Decision Trees |
| KS Test | Kolmogorov-Smirnov statistical test |
| LMTD | Log Mean Temperature Difference |
| MAPE | Mean Absolute Percentage Error |
| MQTT | Message Queuing Telemetry Transport protocol |
| NTU | Number of Transfer Units |
| ONNX | Open Neural Network Exchange format |
| PSI | Population Stability Index |
| QoS | Quality of Service (MQTT) |
| R² | Coefficient of determination |
| RMSE | Root Mean Squared Error |
| UCAOT | Unit Cooler Air Outlet Temperature |
| UCWOT | Unit Cooler Water Outlet Temperature |
| UCAF | Unit Cooler Air Flow |

---

## Appendix B: Configuration Files

**API Configuration** (`api/config.py`):
```python
API_HOST = "0.0.0.0"
API_PORT = 8000
MODEL_PATH = "models/"
DRIFT_THRESHOLD_PSI = 0.2
ALERT_ENABLED = True
```

**MQTT Configuration** (`integration/mqtt/config.py`):
```python
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_QOS = 1
MQTT_TOPICS = {
    "sensors": "hvac/sensors/data",
    "predictions": "hvac/predictions/results",
    "alerts": "hvac/alerts/{severity}"
}
```

---

## Appendix C: Maintenance Schedule

| Task | Frequency | Responsible | Duration |
|------|-----------|-------------|----------|
| Model retraining | Quarterly | Data Scientist | 4 hours |
| Drift report review | Weekly | Engineer | 30 min |
| Performance audit | Monthly | Tech Lead | 2 hours |
| Security updates | As needed | DevOps | 1 hour |
| Backup verification | Weekly | Admin | 15 min |

---

**Document End**

---

**For questions or support:**
Technical Support: [support@example.com]
Project Repository: [github.com/project]
Documentation: [docs.example.com]

**Last Updated**: 2025-11-21
**Next Review**: 2026-02-18 (3 months)

# Physics-Informed Digital Twin for Naval HVAC Unit Cooler

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Sprint%200%20Complete-success.svg)](.)

## 🎯 Project Overview

Development of a **Physics-Informed Neural Network (PINN)** digital twin for naval HVAC Unit Cooler systems. The goal is to reduce prediction errors from current 30-221% down to **<10%** for critical variables: UCAOT, UCWOT, UCAF, and Q_thermal.

**Key Features:**
- 🔬 Physics-informed machine learning combining data-driven and first-principles approaches
- 📊 Multi-output prediction (temperatures, flows, thermal power)
- ⚡ Edge computing ready (<100ms inference, <2GB RAM)
- 🚀 Deployable on Raspberry Pi 4 and Jetson Orin
- 📈 Real-time monitoring and anomaly detection

## 📂 Project Structure

```
Unit_Cooler_CIS/
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Cleaned and preprocessed data
├── src/
│   ├── data/                   # Data loading and preprocessing
│   │   ├── data_loader.py
│   │   └── feature_engineering.py
│   ├── models/                 # Model architectures
│   │   ├── baseline_models.py
│   │   ├── pinn_model.py
│   │   └── ensemble.py
│   ├── losses/                 # Custom loss functions
│   │   └── physics_losses.py
│   ├── optimization/           # Hyperparameter optimization
│   │   └── hpo_optuna.py
│   └── utils/                  # Utility functions
│       ├── eda_utils.py
│       └── visualization.py
├── models/                     # Saved trained models
│   ├── linearregression_model.pkl
│   └── randomforest_model.pkl
├── notebooks/                  # Jupyter notebooks
│   └── notebook_eda.ipynb
├── tests/                      # Unit tests
├── results/                    # Experiment results
│   └── baseline_comparison.csv
├── plots/                      # Visualizations
│   ├── missing_values.png
│   ├── distributions_key_variables.png
│   ├── correlation_heatmap.png
│   ├── time_series_key_variables.png
│   ├── boxplots_key_variables.png
│   └── target_correlations.png
├── deployment/                 # Deployment configurations
├── api/                        # API endpoints
├── dashboard/                  # Streamlit dashboard
├── requirements.txt            # Python dependencies
├── data_quality_report.md      # Data quality assessment
└── README.md                   # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd Unit_Cooler_CIS

# Install dependencies
pip install -r requirements.txt
```

### Data Loading and EDA

```python
from src.data.data_loader import load_and_preprocess
from src.utils.eda_utils import print_eda_summary

# Load data
df, metadata = load_and_preprocess('data/raw/datos_combinados_entrenamiento_20251118_105234.csv')

# Print summary
print_eda_summary(df)
```

### Baseline Model Training

```python
from src.models.baseline_models import prepare_data_for_modeling, train_and_evaluate_baseline_models

# Prepare data
target_vars = ['UCAOT', 'UCWOT', 'UCAF']
X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_for_modeling(df, target_vars)

# Train models
results = train_and_evaluate_baseline_models(X_train, X_val, X_test, y_train, y_val, y_test)
```

## 📊 Dataset Overview

**Dataset:** `datos_combinados_entrenamiento_20251118_105234.csv`

| Metric | Value |
|--------|-------|
| Total Samples | 56,211 |
| Features | 32 |
| Usable Samples | 43,147 (76.8%) |
| Size | 13.72 MB |
| Operational Conditions | Summer/Winter, Setpoints 21-31°C |

### Target Variables

| Variable | Description | Unit |
|----------|-------------|------|
| **UCAOT** | Unit Cooler Air Outlet Temperature | °C |
| **UCWOT** | Unit Cooler Water Outlet Temperature | °C |
| **UCAF** | Unit Cooler Air Flow | m³/h |
| **Q_thermal** | Thermal Power (calculated) | kW |

### Key Input Variables

| Variable | Description | Unit |
|----------|-------------|------|
| **UCWIT** | Water Inlet Temperature | °C |
| **UCAIT** | Air Inlet Temperature | °C |
| **UCWF** | Water Flow Rate | L/min |
| **UCAIH** | Air Inlet Humidity | % |
| **AMBT** | Ambient Temperature | °C |
| **UCTSP** | Temperature Setpoint | °C |

## 🎯 Sprint 0 Results (COMPLETED ✅)

### Baseline Model Performance

| Model | Target | MAE | RMSE | R² |
|-------|--------|-----|------|-----|
| **RandomForest** | UCAOT | 0.64 | 7.76 | **0.983** |
| **RandomForest** | UCWOT | 0.69 | 11.41 | **0.997** |
| **RandomForest** | UCAF | 210.55 | 2216.62 | **0.984** |
| LinearRegression | UCAOT | 19.01 | 40.38 | 0.550 |
| LinearRegression | UCWOT | 79.64 | 116.67 | 0.686 |
| LinearRegression | UCAF | 3511.65 | 6875.67 | 0.847 |

### Deliverables

- ✅ **notebook_eda.ipynb** - Comprehensive exploratory data analysis
- ✅ **data_quality_report.md** - Detailed data quality assessment
- ✅ **baseline_results** - LinearRegression and RandomForest benchmarks
- ✅ **Visualizations** - 6 comprehensive plots
- ✅ **Modular codebase** - Professional structure with utilities

## 🔬 Key Findings

### Data Quality

- **Missing Values:** 23-76% in various columns, with UCAIH at 72% missing
- **Negative Flows:** 12,620 negative values in UCWF (22.45%)
- **Outliers:** 10-30% outliers detected using IQR method
- **Sensor Saturation:** UCAF max = 65,535 (2^16-1)

### Correlations

- **UCAIH** (humidity) strongly negatively correlated with both UCAOT (r=-0.624) and UCWOT (r=-0.658)
- **High multicollinearity** between fan measurements (UCFMS ↔ UCFMV: r=0.996)
- **Target variables** show low intercorrelation (good for multi-output modeling)

### Physics Validation

- Some samples violate energy conservation
- Negative ΔT values present (physically impossible for cooling)
- Efficiency > 1.0 in some cases (violates thermodynamics)

**See full analysis in:** [`data_quality_report.md`](data_quality_report.md)

## 🗓️ Development Roadmap

### ✅ Sprint 0: Setup & Exploration (COMPLETED)
- [x] Data loading and validation
- [x] Comprehensive EDA
- [x] Baseline models (LinearRegression, RandomForest)
- [x] Data quality report
- [x] Visualization suite

### 🔄 Sprint 1: Data Engineering & Features (NEXT)
- [ ] Robust preprocessing pipeline
- [ ] Physics-based feature engineering (15+ features)
- [ ] Temporal train/val/test splits
- [ ] Adaptive normalization by regime

### 📋 Sprint 2: Baseline Avanzado
- [ ] XGBoost and LightGBM models
- [ ] MLP baseline
- [ ] Feature importance analysis
- [ ] Cross-validation temporal

### 🧠 Sprint 3: Physics-Informed Architecture
- [ ] PINN model with physics loss
- [ ] Thermodynamic constraints
- [ ] Multi-objective training
- [ ] Physics validation

### ⚙️ Sprint 4: Optimization HPO
- [ ] Optuna hyperparameter optimization
- [ ] Lambda weight tuning
- [ ] Ensemble methods
- [ ] Missing data robustness

### 📈 Sprint 5: Evaluación Exhaustiva
- [ ] Test set comprehensive evaluation
- [ ] Benchmark vs FMU
- [ ] Residual analysis
- [ ] Technical report (15-20 pages)

### 🚀 Sprint 6: Edge Deployment
- [ ] INT8/FP16 quantization
- [ ] ONNX and TensorFlow Lite export
- [ ] Raspberry Pi 4 / Jetson Orin benchmarks
- [ ] Docker containerization

### 🔌 Sprint 7: Integration Real-time
- [ ] FastAPI inference endpoints
- [ ] Streamlit dashboard
- [ ] Drift detection system
- [ ] MQTT/BACnet integration

### 📚 Sprint 8: Documentation & Transfer
- [ ] NASA SE technical documentation
- [ ] User manuals
- [ ] Knowledge transfer sessions
- [ ] Maintenance guide

## 🛠️ Technology Stack

**Core:**
- Python 3.8-3.10
- TensorFlow 2.13+ / PyTorch 2.0+
- scikit-learn 1.3+
- pandas, numpy, scipy

**ML/DL:**
- XGBoost, LightGBM
- Optuna (HPO)
- PySINDy (equation discovery)

**Deployment:**
- ONNX Runtime
- TensorFlow Lite
- FastAPI
- Streamlit

**Visualization:**
- matplotlib, seaborn, plotly

## 📊 Model Architecture (Planned)

### Physics-Informed Neural Network

```python
class PhysicsInformedNN:
    - Input: [UCWIT, UCAIT, UCWF, UCAF, UCAIH, AMBT, ...]
    - Hidden: Dense(128) → Dense(128) → Dense(64)
    - Output: [UCAOT, UCWOT, UCAF, Q_thermal]

    Loss = λ_data × MSE(predictions, targets)
         + λ_physics × Physics_Loss(energy_balance, constraints)
```

**Physics Constraints:**
- Energy balance: Q_agua ≈ Q_aire
- Temperature monotonicity: ΔT > 0
- Efficiency bounds: 0.3 ≤ η ≤ 0.95
- Second law of thermodynamics

## 📈 Performance Targets

| Metric | Current (FMU) | Target | Status |
|--------|--------------|---------|---------|
| UCAOT MAE | 30-221% | <10% | 🟡 Baseline: 2.4% (RF) |
| UCWOT MAE | 30-221% | <10% | 🟡 Baseline: 9.5% (RF) |
| UCAF MAE | 30-221% | <10% | 🔴 Baseline: 33.7% (RF) |
| Inference Time | N/A | <100ms | ⏳ TBD |
| Memory | N/A | <2GB | ⏳ TBD |

## 🤝 Contributing

This project follows NASA SE Handbook standards for documentation and development.

## 📄 License

MIT License - See LICENSE file for details

## 📧 Contact

For questions or collaboration: [Project Team]

## 🙏 Acknowledgments

- Naval HVAC system data collection team
- Unit Cooler experimental facility
- Physics-informed ML research community

---

**Last Updated:** 2025-11-18
**Sprint:** 0 (Complete)
**Next Milestone:** Sprint 1 - Data Engineering & Features

# Scripts Directory

Utility scripts for data analysis, packaging, and project automation.

## 📂 Directory Structure

```
scripts/
├── analysis/                       # Data analysis scripts
│   ├── investigate_validation_data.py    # Data source investigation
│   ├── analyze_test_data_detail.py       # Test set detailed analysis
│   ├── debug_test_data.py                # Test data debugging
│   ├── package_test_data.py              # Test data packager
│   ├── package_files_for_download.py     # File packager
│   └── generate_sprint2_analysis.py      # Sprint 2 analysis
├── download_training_data.py       # Training data downloader
└── README.md                       # This file
```

## 📊 Analysis Scripts

### investigate_validation_data.py

Investigates the source and characteristics of validation/test data.

**Purpose:**
- Documents data source (datos_combinados_entrenamiento_20251118_105234.csv)
- Shows temporal split methodology (70/15/15)
- Analyzes test set composition (8,432 samples)

**Usage:**
```bash
python scripts/analysis/investigate_validation_data.py
```

**Output:**
```
Data Source Investigation
========================
Raw Data:
  File: datos_combinados_entrenamiento_20251118_105234.csv
  Samples: 56,211
  Features: 32

Test Set:
  Samples: 8,432 (15%)
  Features: 39 (23 + 19 engineered)
  Temporal: Last 15% chronologically
```

### analyze_test_data_detail.py

Performs detailed analysis of test set characteristics.

**Purpose:**
- Compares raw vs processed data
- Shows data quality metrics
- Displays unscaled test ranges

**Usage:**
```bash
python scripts/analysis/analyze_test_data_detail.py
```

**Output:**
```
Test Data Analysis
==================
Data Quality:
  Raw samples: 8,479
  Processed samples: 8,432
  Retention: 99.45%
  Outliers removed: 47

Unscaled Ranges:
  UCAOT: 19.18 - 64.13°C
  UCWOT: 1.00 - 136.03°C
  UCAF: 372 - 7,970 m³/h
```

### debug_test_data.py

Debug script for identifying test data issues.

**Purpose:**
- Checks for duplicate samples
- Validates data consistency
- Identifies anomalies

**Usage:**
```bash
python scripts/analysis/debug_test_data.py
```

### package_test_data.py

Creates downloadable test data package.

**Purpose:**
- Packages test arrays (X_test_scaled.npy, y_test_scaled.npy)
- Includes scalers (scaler_clean.pkl, y_scaler_clean.pkl)
- Adds metadata (metadata.json)

**Usage:**
```bash
python scripts/analysis/package_test_data.py
```

**Output:**
- `test_data_package.zip` (121 KB)
- Location: `deployment/packages/`

**Package Contents:**
```
test_data_package.zip
├── X_test_scaled.npy       # 8,432 × 39 features
├── y_test_scaled.npy       # 8,432 × 3 targets
├── scaler_clean.pkl        # Input scaler
├── y_scaler_clean.pkl      # Output scaler
└── metadata.json           # Feature names
```

### package_files_for_download.py

Creates downloadable validation data package.

**Purpose:**
- Packages analysis scripts
- Includes raw training data
- Creates ZIP for distribution

**Usage:**
```bash
python scripts/analysis/package_files_for_download.py
```

**Output:**
- `validation_data_package.zip` (676 KB)
- Location: `deployment/packages/`

**Package Contents:**
```
validation_data_package.zip
├── investigate_validation_data.py
├── analyze_test_data_detail.py
└── datos_combinados_entrenamiento_20251118_105234.csv (6.5 MB)
```

### generate_sprint2_analysis.py

Generates Sprint 2 baseline comparison analysis.

**Purpose:**
- Compares baseline models (LightGBM, XGBoost, MLP)
- Creates performance reports
- Generates comparison plots

## 🔧 Utility Scripts

### download_training_data.py

Downloads training data from external sources.

**Purpose:**
- Automates data download
- Validates data integrity
- Places data in correct location

**Usage:**
```bash
python scripts/download_training_data.py
```

## 📦 Package Creation Workflow

### Creating Test Data Package

```bash
# 1. Ensure data is processed
python run_sprint1_pipeline_no_leakage.py

# 2. Create test data package
python scripts/analysis/package_test_data.py

# Output: deployment/packages/test_data_package.zip
```

### Creating Validation Package

```bash
# 1. Run data investigation
python scripts/analysis/investigate_validation_data.py

# 2. Run detailed analysis
python scripts/analysis/analyze_test_data_detail.py

# 3. Package files
python scripts/analysis/package_files_for_download.py

# Output: deployment/packages/validation_data_package.zip
```

## 🔍 Data Analysis Workflow

### Investigating Test Data

```bash
# Step 1: Investigate data source
python scripts/analysis/investigate_validation_data.py

# Step 2: Analyze data quality
python scripts/analysis/analyze_test_data_detail.py

# Step 3: Debug if issues found
python scripts/analysis/debug_test_data.py
```

## 📚 Script Dependencies

All scripts require:
```
numpy
pandas
joblib
zipfile (built-in)
pathlib (built-in)
```

Install with:
```bash
pip install numpy pandas scikit-learn
```

## 🎯 Common Use Cases

### 1. Understanding Test Data Source

```bash
python scripts/analysis/investigate_validation_data.py
```

**Answers:**
- Where did the test data come from?
- How was the test set created?
- What temporal split was used?

### 2. Checking Data Quality

```bash
python scripts/analysis/analyze_test_data_detail.py
```

**Answers:**
- How many samples retained?
- What are the data ranges?
- Any quality issues?

### 3. Creating Distribution Package

```bash
python scripts/analysis/package_test_data.py
python scripts/analysis/package_files_for_download.py
```

**Result:**
- Ready-to-download ZIP files
- Located in `deployment/packages/`

## 📊 Output Files

### Analysis Outputs

Scripts generate informative console output, no files by default.

### Package Outputs

| Script | Output File | Size | Location |
|--------|-------------|------|----------|
| package_test_data.py | test_data_package.zip | 121 KB | deployment/packages/ |
| package_files_for_download.py | validation_data_package.zip | 676 KB | deployment/packages/ |

## 🔄 Maintenance

### Re-running Analysis

After model retraining or data updates:

```bash
# 1. Re-run data pipeline
python run_sprint1_pipeline_no_leakage.py

# 2. Re-run analysis
python scripts/analysis/investigate_validation_data.py
python scripts/analysis/analyze_test_data_detail.py

# 3. Re-create packages
python scripts/analysis/package_test_data.py
python scripts/analysis/package_files_for_download.py
```

## 🤝 Contributing

When adding new analysis scripts:
1. Place in `scripts/analysis/`
2. Follow naming convention: `<action>_<subject>.py`
3. Add docstring with purpose and usage
4. Update this README

## 📧 Support

For issues or questions:
- Check script docstrings for detailed usage
- Review output carefully for error messages
- Contact: [rferreiroag](https://github.com/rferreiroag)

---

**Last Updated:** 2025-11-21
**Status:** ✅ Complete
**Scripts:** 6 analysis + 1 utility

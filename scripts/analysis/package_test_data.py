"""
Package test data files for download
Creates a ZIP with test data arrays and scalers
"""
import zipfile
import os
from pathlib import Path

print("="*80)
print(" PACKAGING TEST DATA FILES")
print("="*80)

# Files to include
files_to_package = [
    ('data/processed_no_leakage/X_test_scaled.npy', 'X_test_scaled.npy'),
    ('data/processed_no_leakage/y_test_scaled.npy', 'y_test_scaled.npy'),
    ('data/processed_no_leakage/scaler_clean.pkl', 'scaler_clean.pkl'),
    ('data/processed_no_leakage/y_scaler_clean.pkl', 'y_scaler_clean.pkl'),
    ('data/processed_no_leakage/metadata.json', 'metadata.json')
]

# Create ZIP file
zip_filename = 'test_data_package.zip'

print(f"\n[1/2] Creating ZIP package: {zip_filename}")
print("-"*80)

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for source_path, archive_name in files_to_package:
        if os.path.exists(source_path):
            file_size = os.path.getsize(source_path) / (1024 * 1024)  # MB
            print(f"  [OK] Adding: {archive_name} ({file_size:.2f} MB)")
            zipf.write(source_path, archive_name)
        else:
            print(f"  [SKIP] Not found: {source_path}")

# Get ZIP file size
zip_size = os.path.getsize(zip_filename) / (1024 * 1024)  # MB

print("\n[2/2] Package created successfully!")
print("-"*80)
print(f"\nPackage details:")
print(f"  Filename: {zip_filename}")
print(f"  Size: {zip_size:.2f} MB")
print(f"  Location: {os.path.abspath(zip_filename)}")

# List contents
print(f"\nContents:")
with zipfile.ZipFile(zip_filename, 'r') as zipf:
    for info in zipf.filelist:
        print(f"  - {info.filename} ({info.file_size / (1024*1024):.2f} MB)")

print("\n" + "="*80)
print(" TEST DATA PACKAGE READY")
print("="*80)
print(f"\nDownload: {os.path.abspath(zip_filename)}")
print("\nContents:")
print("  1. X_test_scaled.npy - Test features (8,432 samples x 39 features)")
print("  2. y_test_scaled.npy - Test targets (8,432 samples x 3 outputs)")
print("  3. scaler_clean.pkl - StandardScaler for unscaling X_test")
print("  4. y_scaler_clean.pkl - StandardScaler for unscaling y_test")
print("  5. metadata.json - Feature and target names")
print("\nUsage example:")
print("  import numpy as np")
print("  import joblib")
print("  X_test_scaled = np.load('X_test_scaled.npy')")
print("  y_test_scaled = np.load('y_test_scaled.npy')")
print("  scaler_X = joblib.load('scaler_clean.pkl')")
print("  scaler_y = joblib.load('y_scaler_clean.pkl')")
print("  X_test = scaler_X.inverse_transform(X_test_scaled)")
print("  y_test = scaler_y.inverse_transform(y_test_scaled)")
print("="*80)

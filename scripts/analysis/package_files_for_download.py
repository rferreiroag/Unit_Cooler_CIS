"""
Package files for download
Creates a ZIP file with requested analysis scripts and data
"""
import zipfile
import os
from pathlib import Path

print("="*80)
print(" PACKAGING FILES FOR DOWNLOAD")
print("="*80)

# Files to include
files_to_package = [
    ('analyze_test_data_detail.py', 'analyze_test_data_detail.py'),
    ('investigate_validation_data.py', 'investigate_validation_data.py'),
    ('data/raw/datos_combinados_entrenamiento_20251118_105234.csv',
     'datos_combinados_entrenamiento_20251118_105234.csv')
]

# Create ZIP file
zip_filename = 'validation_data_package.zip'

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
        print(f"  - {info.filename} ({info.file_size / 1024:.2f} KB)")

print("\n" + "="*80)
print(" PACKAGE READY FOR DOWNLOAD")
print("="*80)
print(f"\nDownload: {os.path.abspath(zip_filename)}")
print("\nContents:")
print("  1. analyze_test_data_detail.py - Detailed test data analysis")
print("  2. investigate_validation_data.py - Data source investigation")
print("  3. datos_combinados_entrenamiento_20251118_105234.csv - Raw HVAC data (6.5 MB)")
print("="*80)

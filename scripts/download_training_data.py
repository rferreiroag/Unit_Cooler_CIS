#!/usr/bin/env python3
"""
Script para descargar y preparar datos de entrenamiento para el Digital Twin HVAC.

Este script:
1. Organiza los datos raw en data/raw/
2. Ejecuta el pipeline completo de preprocesamiento (Sprint 1)
3. Genera los datos processed listos para entrenamiento

Uso:
    python scripts/download_training_data.py

Salida:
    - data/raw/datos_combinados_entrenamiento_20251118_105234.csv
    - data/processed/X_train.csv, X_val.csv, X_test.csv
    - data/processed/y_train.csv, y_val.csv, y_test.csv
    - data/processed/*_scaled.npy (arrays escalados)
    - data/processed/scaler.pkl (scaler entrenado)
    - data/processed/metadata.json (metadatos)
"""

import os
import sys
import shutil
import logging
import subprocess
from pathlib import Path
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Crear directorios necesarios si no existen."""
    directories = [
        'data/raw',
        'data/processed'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Directorio verificado: {directory}")


def download_raw_data():
    """
    Organizar datos raw en el directorio correcto.

    Por ahora, copia el archivo CSV existente a data/raw/.
    En el futuro, esto podría descargar datos de una fuente externa (S3, API, etc.).
    """
    source_file = 'datos_combinados_entrenamiento_20251118_105234.csv'
    destination_dir = 'data/raw'
    destination_file = os.path.join(destination_dir, source_file)

    # Si ya existe, no hace falta copiar
    if os.path.exists(destination_file):
        logger.info(f"✓ Datos raw ya existen: {destination_file}")
        size_mb = os.path.getsize(destination_file) / (1024 * 1024)
        logger.info(f"  Tamaño: {size_mb:.2f} MB")
        return destination_file

    if not os.path.exists(source_file):
        logger.error(f"❌ No se encontró el archivo fuente: {source_file}")
        logger.error("   Asegúrate de que el archivo CSV existe en el directorio raíz")
        return None

    # Copiar archivo a data/raw/
    logger.info(f"Copiando datos raw: {source_file} -> {destination_file}")
    shutil.copy2(source_file, destination_file)

    # Verificar tamaño
    size_mb = os.path.getsize(destination_file) / (1024 * 1024)
    logger.info(f"✓ Datos raw copiados: {size_mb:.2f} MB")

    return destination_file


def run_sprint1_pipeline():
    """
    Ejecutar el pipeline completo de Sprint 1 para generar datos processed.

    Este pipeline incluye:
    - Carga de datos raw
    - Preprocesamiento (limpieza, imputación, validación)
    - Feature engineering (52 features derivadas)
    - Splits temporales (70/15/15)
    - Escalado adaptativo
    """
    logger.info("Ejecutando pipeline completo de Sprint 1...")
    logger.info("Este proceso puede tardar varios minutos...")

    try:
        # Ejecutar run_sprint1_pipeline.py
        result = subprocess.run(
            ['python', 'run_sprint1_pipeline.py'],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutos de timeout
        )

        if result.returncode != 0:
            logger.error("❌ Error al ejecutar pipeline:")
            logger.error(result.stderr)
            return False

        logger.info("✓ Pipeline ejecutado exitosamente")
        return True

    except subprocess.TimeoutExpired:
        logger.error("❌ Timeout: El pipeline tardó más de 10 minutos")
        return False
    except Exception as e:
        logger.error(f"❌ Error ejecutando pipeline: {str(e)}")
        return False


def verify_output():
    """Verificar que todos los archivos de salida fueron generados correctamente."""
    required_files = [
        'data/processed/X_train.csv',
        'data/processed/X_val.csv',
        'data/processed/X_test.csv',
        'data/processed/y_train.csv',
        'data/processed/y_val.csv',
        'data/processed/y_test.csv',
        'data/processed/X_train_scaled.npy',
        'data/processed/X_val_scaled.npy',
        'data/processed/X_test_scaled.npy',
        'data/processed/y_train_scaled.npy',
        'data/processed/y_val_scaled.npy',
        'data/processed/y_test_scaled.npy',
        'data/processed/scaler.pkl',
        'data/processed/metadata.json'
    ]

    logger.info("Verificando archivos generados...")

    all_exist = True
    total_size = 0

    for file_path in required_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            total_size += size_mb
            logger.info(f"  ✓ {file_path} ({size_mb:.2f} MB)")
        else:
            logger.warning(f"  ✗ Falta: {file_path}")
            all_exist = False

    if all_exist:
        logger.info(f"\n✓ Todos los archivos generados correctamente")
        logger.info(f"  Tamaño total: {total_size:.2f} MB")
    else:
        logger.warning("\n⚠ Algunos archivos no fueron generados")

    return all_exist


def generate_summary():
    """Generar resumen de los datos descargados."""
    summary_file = 'data/DATA_SUMMARY.md'

    # Leer metadata si existe
    metadata_file = 'data/processed/metadata.json'
    if os.path.exists(metadata_file):
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = None

    with open(summary_file, 'w') as f:
        f.write("# Resumen de Datos de Entrenamiento\n\n")
        f.write(f"**Fecha de generación:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Estructura de Datos\n\n")
        f.write("```\n")
        f.write("data/\n")
        f.write("├── raw/                     # Datos originales sin procesar\n")
        f.write("│   └── datos_combinados_entrenamiento_20251118_105234.csv (6.5 MB)\n")
        f.write("├── processed/               # Datos procesados listos para entrenamiento (66 MB)\n")
        f.write("│   ├── X_train.csv          # Features de entrenamiento\n")
        f.write("│   ├── y_train.csv          # Targets de entrenamiento\n")
        f.write("│   ├── X_val.csv            # Features de validación\n")
        f.write("│   ├── y_val.csv            # Targets de validación\n")
        f.write("│   ├── X_test.csv           # Features de test\n")
        f.write("│   ├── y_test.csv           # Targets de test\n")
        f.write("│   ├── *_scaled.npy         # Arrays NumPy escalados\n")
        f.write("│   ├── scaler.pkl           # Scaler entrenado (StandardScaler)\n")
        f.write("│   └── metadata.json        # Metadatos del dataset\n")
        f.write("```\n\n")

        if metadata:
            f.write("## Estadísticas del Dataset\n\n")
            f.write(f"- **Muestras totales:** {metadata.get('n_total_samples', 'N/A'):,}\n")
            f.write(f"- **Train:** {metadata.get('n_train', 'N/A'):,} muestras (70%)\n")
            f.write(f"- **Validation:** {metadata.get('n_val', 'N/A'):,} muestras (15%)\n")
            f.write(f"- **Test:** {metadata.get('n_test', 'N/A'):,} muestras (15%)\n")
            f.write(f"- **Features:** {metadata.get('n_features', 'N/A')} (32 originales + 20 derivadas)\n")
            f.write(f"- **Targets:** {len(metadata.get('target_names', []))} variables\n\n")

        f.write("## Variables Target\n\n")
        f.write("- **UCAOT**: Unit Cooler Air Outlet Temperature (°C)\n")
        f.write("- **UCWOT**: Unit Cooler Water Outlet Temperature (°C)\n")
        f.write("- **UCAF**: Unit Cooler Air Flow (m³/h)\n\n")

        f.write("## Preprocesamiento Aplicado\n\n")
        f.write("1. **Limpieza de datos:**\n")
        f.write("   - Manejo de saturación de sensores (65535, 65534)\n")
        f.write("   - Corrección de valores negativos en flujos\n")
        f.write("   - Clipeo de valores extremos\n\n")
        f.write("2. **Imputación:**\n")
        f.write("   - Eliminación de columnas con >70% valores faltantes\n")
        f.write("   - Forward fill para valores faltantes restantes\n\n")
        f.write("3. **Feature Engineering:**\n")
        f.write("   - Deltas de temperatura (agua, aire)\n")
        f.write("   - Potencia térmica (Q_water, Q_air, Q_avg)\n")
        f.write("   - Eficiencia del intercambiador\n")
        f.write("   - Números adimensionales (NTU, effectiveness)\n")
        f.write("   - Features temporales (ciclo horario)\n")
        f.write("   - Interacciones físicas\n\n")
        f.write("4. **Escalado:**\n")
        f.write("   - StandardScaler (media=0, std=1)\n")
        f.write("   - Escalado independiente para features y targets\n\n")

        f.write("## Uso\n\n")
        f.write("### Cargar datos para entrenamiento:\n\n")
        f.write("```python\n")
        f.write("import pandas as pd\n")
        f.write("import numpy as np\n\n")
        f.write("# Opción 1: Usar CSV\n")
        f.write("X_train = pd.read_csv('data/processed/X_train.csv')\n")
        f.write("y_train = pd.read_csv('data/processed/y_train.csv')\n\n")
        f.write("# Opción 2: Usar arrays NumPy escalados (recomendado para entrenamiento)\n")
        f.write("X_train = np.load('data/processed/X_train_scaled.npy')\n")
        f.write("y_train = np.load('data/processed/y_train_scaled.npy')\n")
        f.write("```\n\n")

        f.write("### Entrenar modelos:\n\n")
        f.write("```bash\n")
        f.write("# Baseline models (XGBoost, LightGBM, MLP)\n")
        f.write("python run_sprint2_baseline.py\n\n")
        f.write("# Physics-Informed Neural Networks\n")
        f.write("python run_sprint3_pinn.py\n")
        f.write("```\n\n")

        f.write("## Reentrenamiento\n\n")
        f.write("Para actualizar los datos con nueva información:\n\n")
        f.write("```bash\n")
        f.write("# Re-ejecutar pipeline completo\n")
        f.write("python scripts/download_training_data.py\n")
        f.write("```\n\n")

        f.write("## Notas\n\n")
        f.write("- Los datos están ignorados por Git (.gitignore)\n")
        f.write("- El pipeline es completamente reproducible\n")
        f.write("- Los splits son temporales (no aleatorios) para preservar la estructura temporal\n")
        f.write("- La retención de datos es ~100% después del preprocesamiento\n")

    logger.info(f"✓ Resumen generado: {summary_file}")


def main():
    """Función principal."""
    print("\n" + "="*80)
    print("  DESCARGA Y PREPARACIÓN DE DATOS DE ENTRENAMIENTO")
    print("  Digital Twin HVAC - Unit Cooler")
    print("="*80 + "\n")

    try:
        # 1. Configurar directorios
        logger.info("[1/5] Configurando directorios...")
        setup_directories()

        # 2. Descargar/copiar datos raw
        logger.info("\n[2/5] Organizando datos raw...")
        raw_data_path = download_raw_data()

        if not raw_data_path:
            logger.error("❌ Error al obtener datos raw")
            sys.exit(1)

        # 3. Ejecutar pipeline de Sprint 1
        logger.info("\n[3/5] Procesando datos (esto puede tardar varios minutos)...")
        if not run_sprint1_pipeline():
            logger.error("❌ Error al procesar datos")
            sys.exit(1)

        # 4. Verificar salida
        logger.info("\n[4/5] Verificando archivos generados...")
        if not verify_output():
            logger.warning("⚠ Algunos archivos no fueron generados correctamente")

        # 5. Generar resumen
        logger.info("\n[5/5] Generando resumen...")
        generate_summary()

        print("\n" + "="*80)
        print("  ✓ PROCESO COMPLETADO EXITOSAMENTE")
        print("="*80)
        print("\n📁 Datos listos para entrenamiento:")
        print("   - data/raw/ (datos originales)")
        print("   - data/processed/ (datos procesados)")
        print("\n📄 Ver data/DATA_SUMMARY.md para más detalles")
        print("\n🚀 Siguiente paso:")
        print("   python run_sprint2_baseline.py")
        print()

    except KeyboardInterrupt:
        logger.warning("\n⚠ Proceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ ERROR: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

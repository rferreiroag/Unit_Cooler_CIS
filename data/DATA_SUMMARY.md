# Resumen de Datos de Entrenamiento

**Fecha de generación:** 2025-11-20 22:04:00

## Estructura de Datos

```
data/
├── raw/                     # Datos originales sin procesar
│   └── datos_combinados_entrenamiento_20251118_105234.csv (6.5 MB)
├── processed/               # Datos procesados listos para entrenamiento (66 MB)
│   ├── X_train.csv          # Features de entrenamiento
│   ├── y_train.csv          # Targets de entrenamiento
│   ├── X_val.csv            # Features de validación
│   ├── y_val.csv            # Targets de validación
│   ├── X_test.csv           # Features de test
│   ├── y_test.csv           # Targets de test
│   ├── *_scaled.npy         # Arrays NumPy escalados
│   ├── scaler.pkl           # Scaler entrenado (StandardScaler)
│   └── metadata.json        # Metadatos del dataset
```

## Estadísticas del Dataset

- **Muestras totales:** 56,211
- **Train:** 39,347 muestras (70%)
- **Validation:** 8,432 muestras (15%)
- **Test:** 8,432 muestras (15%)
- **Features:** 52 (23 originales + 29 derivadas)
- **Targets:** 3 variables (UCAOT, UCWOT, UCAF)

## Uso

### Cargar datos:

\`\`\`python
import pandas as pd
import numpy as np

# CSV (sin escalar)
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv')

# NumPy escalados (RECOMENDADO)
X_train = np.load('data/processed/X_train_scaled.npy')
y_train = np.load('data/processed/y_train_scaled.npy')
\`\`\`

### Entrenar modelos:

\`\`\`bash
python run_sprint2_baseline.py
\`\`\`

## Reentrenamiento

\`\`\`bash
python scripts/download_training_data.py
\`\`\`

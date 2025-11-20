# FMU con Entradas de Sensores (Sin Data Leakage)

## Descripción

Esta FMU acepta **solo 20 variables de sensores** como entrada y predice 3 salidas (UCAOT, UCWOT, UCAF).

**Diferencia clave**: La FMU calcula internamente las 19 features derivadas, sin requerir los targets como entrada. Esto garantiza que funcione en producción en tiempo real.

## Especificaciones

### Entradas (20 sensores)

```
1.  AMBT   - Ambient Temperature (°C)
2.  UCTSP  - Unit Cooler Temperature Setpoint (°C)
3.  CPSP   - Chilled Plant Setpoint
4.  UCAIT  - Unit Cooler Air Inlet Temperature (°C)
5.  CPPR   - Chilled Plant Pressure (bar)
6.  UCWF   - Unit Cooler Water Flow (L/min)
7.  CPMC   - Chilled Plant Motor Current (A)
8.  MVDP   - Mixing Valve Differential Pressure
9.  CPCF   - Chilled Plant Condenser Flow
10. UCFS   - Unit Cooler Fan Speed (%)
11. MVCV   - Mixing Valve Control Value
12. UCHV   - Unit Cooler Heating Valve
13. CPMV   - Chilled Plant Main Valve
14. UCHC   - Unit Cooler Heating Current
15. UCWIT  - Unit Cooler Water Inlet Temperature (°C)
16. UCFMS  - Unit Cooler Fan Motor Speed
17. CPDP   - Chilled Plant Differential Pressure
18. UCWDP  - Unit Cooler Water Differential Pressure
19. MVWF   - Mixing Valve Water Flow
20. UCOM   - Unit Cooler Operation Mode
```

### Salidas (3 predicciones)

```
1. UCAOT - Unit Cooler Air Outlet Temperature (°C)
2. UCWOT - Unit Cooler Water Outlet Temperature (°C)
3. UCAF  - Unit Cooler Air Flow (m³/h)
```

### Procesamiento Interno

La FMU computa internamente 19 features derivadas:

1. **Temperaturas** (5): T_approach, T_water_ambient_diff, T_air_ambient_diff, setpoint_inlet_diff, setpoint_ambient_diff
2. **Flujos** (1): mdot_water
3. **Térmica** (2): C_water, Q_max_water
4. **Potencia** (3): P_fan_estimate, P_pump_estimate, P_total_estimate
5. **Temporales** (4): time_index, cycle_hour, hour_sin, hour_cos
6. **Interacciones** (4): T_water_x_flow, ambient_x_inlet, setpoint_x_flow, T_water_x_pressure

**Total**: 39 features (20 sensores + 19 derivadas)

## Rendimiento

- **UCAOT**: R² = 0.91, MAE = 0.14°C
- **UCWOT**: R² = 0.75, MAE = 0.25°C
- **UCAF**: R² = 0.75, MAE = 0.20 m³/h

**R² promedio**: 0.80 (realista y funcional en producción)

## Archivos FMU

- `HVACUnitCoolerFMU.fmu` (0.64 MB) - FMU con sensores (RECOMENDADA)
- `HVACUnitCooler.fmu` (2.78 MB) - FMU legacy (no usar)

## Uso

### Modelica/Dymola

```modelica
model TestHVACFMU
  HVACUnitCoolerFMU fmu;
equation
  // Conectar sensores
  fmu.AMBT = sensors.ambient_temp;
  fmu.UCTSP = control.setpoint;
  fmu.UCWF = sensors.water_flow;
  // ... (conectar los 20 sensores)

  // Leer predicciones
  predictions.air_out_temp = fmu.UCAOT;
  predictions.water_out_temp = fmu.UCWOT;
  predictions.air_flow = fmu.UCAF;
end TestHVACFMU;
```

### MATLAB/Simulink

1. Añadir bloque FMU Import
2. Seleccionar `HVACUnitCoolerFMU.fmu`
3. Conectar 20 entradas de sensores
4. Leer 3 salidas de predicción

### Python (FMPy)

```python
from fmpy import simulate_fmu
import numpy as np

# Valores de sensores
sensor_inputs = {
    'AMBT': 23.0,
    'UCTSP': 21.0,
    'UCAIT': 22.0,
    'UCWIT': 23.5,
    'UCWF': 120.0,
    'UCFS': 75.0,
    # ... resto de sensores
}

# Simular FMU
result = simulate_fmu(
    'deployment/fmu/HVACUnitCoolerFMU.fmu',
    start_values=sensor_inputs,
    stop_time=1.0
)

print(f"UCAOT (Air Out Temp): {result['UCAOT'][-1]:.2f}°C")
print(f"UCWOT (Water Out Temp): {result['UCWOT'][-1]:.2f}°C")
print(f"UCAF (Air Flow): {result['UCAF'][-1]:.2f} m³/h")
```

### Python (Standalone test)

```python
import joblib
import numpy as np

# Cargar modelo
model_data = joblib.load('models/lightgbm_model_no_leakage.pkl')
models = model_data['models']

# Cargar scaler
scaler = joblib.load('data/processed_no_leakage/scaler.pkl')

# Valores de prueba (20 sensores + 19 derivadas = 39)
features = np.array([...])  # 39 features

# Escalar
features_scaled = scaler.transform(features.reshape(1, -1))

# Predecir
ucaot = models['UCAOT'].predict(features_scaled)[0]
ucwot = models['UCWOT'].predict(features_scaled)[0]
ucaf = models['UCAF'].predict(features_scaled)[0]
```

## Ventajas vs FMU Original

| Aspecto | FMU Original | FMU Nueva (Sensores) |
|---------|-------------|---------------------|
| Entradas | 52 (con leakage) | 20 (solo sensores) ✅ |
| Data leakage | ❌ Sí (17 features) | ✅ No |
| Funciona en producción | ❌ No | ✅ Sí |
| R² | 0.99 (irreal) | 0.80 (realista) |
| Tamaño | - | 0.64 MB |

## Regenerar FMU

Si necesitas regenerar la FMU con cambios:

```bash
# 1. Reentrenar modelo (opcional)
python train_model_no_leakage.py

# 2. Exportar FMU
python export_fmu_sensor_inputs.py
```

## Validación

Para validar que la FMU funciona correctamente:

```bash
# Instalar FMPy
pip install fmpy

# Simular FMU
fmpy simulate deployment/fmu/HVACUnitCoolerFMU.fmu --show-plot
```

## Estructura de Archivos

```
deployment/fmu/
├── HVACUnitCoolerFMU.fmu           # FMU lista para usar ✅
├── hvac_fmu_sensor_inputs.py       # Código fuente de la FMU
├── resources/                      # Recursos embebidos
│   ├── lightgbm_model_no_leakage.pkl
│   ├── scaler.pkl
│   └── metadata.json
└── README_SENSOR_INPUTS.md         # Este archivo
```

## Troubleshooting

### Error: "Model file not found"
Solución: Ejecuta `python train_model_no_leakage.py` primero

### Error: "pythonfmu not found"
Solución: `pip install pythonfmu`

### FMU no genera salidas correctas
- Verifica que las 20 entradas de sensores estén conectadas
- Comprueba rangos de valores razonables (ej: temperaturas 15-35°C)

## Referencias

- **Modelo**: LightGBM sin data leakage
- **Datos**: data/processed_no_leakage/
- **Changelog**: CHANGELOG_NO_LEAKAGE.md
- **Script de entrenamiento**: train_model_no_leakage.py
- **Script de exportación**: export_fmu_sensor_inputs.py

## Contacto

Para reportar problemas o preguntas sobre la FMU, consultar la documentación del proyecto.

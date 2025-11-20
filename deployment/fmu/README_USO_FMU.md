# Cómo Usar el FMU: HVACUnitCooler.fmu

## 📦 Archivo Generado

```
deployment/fmu/HVACUnitCooler.fmu (2.9 MB)
```

**Contenido:**
- ✅ Binario Linux 64-bit (`HVACUnitCooler.so` - 346 KB)
- ✅ Binario Windows 64-bit (`HVACUnitCooler.dll` - 256 KB)
- ✅ Modelos LightGBM embebidos (2.2 MB)
- ✅ StandardScaler para preprocesamiento
- ✅ FMI 2.0 Co-Simulation standard

---

## 🎯 Características del FMU

### Entradas (52 features)

**Sensores Raw (20):**
- AMBT, UCTSP, CPSP, UCAIT, CPPR, UCWF, CPMC, MVDP, CPCF, UCFS
- MVCV, UCHV, CPMV, UCHC, UCWIT, UCFMS, CPDP, UCWDP, MVWF, UCOM

**Features de Temperatura (5):**
- delta_T_water, delta_T_air, T_approach, T_water_avg, T_air_avg

**Features de Potencia Térmica (7):**
- mdot_water, mdot_air, Q_water, Q_air, Q_avg, Q_imbalance, Q_imbalance_pct

**Features de Intercambiador de Calor (4):**
- efficiency_HX, effectiveness, NTU, C_ratio

**Features de Dinámica de Fluidos (2):**
- Re_air_estimate, flow_ratio

**Features de Control (3):**
- delta_T_ratio, setpoint_error, setpoint_error_abs

**Features de Potencia & Eficiencia (4):**
- P_fan_estimate, P_pump_estimate, P_total_estimate, COP_estimate

**Features Temporales (5):**
- time_index, cycle_hour, hour_sin, hour_cos

**Features de Interacción (3):**
- T_water_x_flow, T_air_x_flow, ambient_x_inlet

### Salidas (3 predictions)

- **UCAOT**: Unit Cooler Air Outlet Temperature (°C) - R²=0.993
- **UCWOT**: Unit Cooler Water Outlet Temperature (°C) - R²=0.998
- **UCAF**: Unit Cooler Air Flow - R²=1.000

---

## 🚀 Uso en Diferentes Herramientas

### 1. OpenModelica

```modelica
model TestHVACUnitCooler
  FMU.HVACUnitCooler fmu(
    AMBT=25.0,
    UCTSP=20.0,
    CPSP=10.0,
    UCAIT=25.0
    // ... resto de 48 inputs
  );
equation
  // Conectar entradas
  fmu.AMBT = ambient_temperature.y;
  fmu.UCTSP = setpoint.y;

  // Leer salidas
  air_outlet_temp = fmu.UCAOT;
  water_outlet_temp = fmu.UCWOT;
  air_flow = fmu.UCAF;
end TestHVACUnitCooler;
```

**Comandos:**
```bash
# Importar FMU en OpenModelica
OMEdit -> File -> Import FMU -> Seleccionar HVACUnitCooler.fmu

# O via comando:
omc> importFMU("HVACUnitCooler.fmu", "1.0")
```

---

### 2. MATLAB/Simulink

**Opción A: Simulink FMU Import Block**

1. Instalar FMU Import toolbox:
   ```matlab
   % En MATLAB R2018a+, el soporte FMU está incluido
   ```

2. En Simulink:
   - Add Block → Simulink → FMU Import
   - Seleccionar `HVACUnitCooler.fmu`
   - Configurar entradas (52 signals)
   - Leer salidas (3 signals)

**Opción B: FMPy desde MATLAB**

```matlab
% Instalar FMPy
!pip install fmpy

% Simular FMU
fmu_path = 'deployment/fmu/HVACUnitCooler.fmu';
result = fmpy.simulate(fmu_path, ...
    'start_time', 0, ...
    'stop_time', 100, ...
    'step_size', 1, ...
    'start_values', struct(...
        'AMBT', 25.0, ...
        'UCTSP', 20.0, ...
        'CPSP', 10.0 ...
        % ... resto de inputs
    ));

% Plotear resultados
plot(result.time, result.UCAOT);
xlabel('Time (s)');
ylabel('UCAOT (°C)');
```

---

### 3. Dymola

```modelica
// Importar FMU
File → Import → FMU → Select HVACUnitCooler.fmu

// Usar en modelo
model HVACSystem
  HVACUnitCooler.HVACUnitCooler cooler;
equation
  cooler.AMBT = building.ambient_temp;
  cooler.UCTSP = controller.setpoint;
  // ...
end HVACSystem;
```

---

### 4. Python (FMPy) - Testing & Validation

**Instalación:**
```bash
pip install fmpy
```

**Simulación:**
```python
from fmpy import simulate_fmu
import numpy as np
import matplotlib.pyplot as plt

# Simular FMU
result = simulate_fmu(
    'deployment/fmu/HVACUnitCooler.fmu',
    start_time=0.0,
    stop_time=100.0,
    step_size=1.0,
    start_values={
        'AMBT': 25.0,
        'UCTSP': 20.0,
        'CPSP': 10.0,
        'UCAIT': 25.0,
        'CPPR': 2.0,
        'UCWF': 1.0,
        'CPMC': 50.0,
        'MVDP': 0.5,
        'CPCF': 1.5,
        'UCFS': 1500.0,
        # ... (resto de 42 inputs con valores por defecto)
    },
    output=['UCAOT', 'UCWOT', 'UCAF']
)

# Plotear resultados
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(result['time'], result['UCAOT'])
plt.xlabel('Time (s)')
plt.ylabel('UCAOT (°C)')
plt.title('Air Outlet Temperature')

plt.subplot(1, 3, 2)
plt.plot(result['time'], result['UCWOT'])
plt.xlabel('Time (s)')
plt.ylabel('UCWOT (°C)')
plt.title('Water Outlet Temperature')

plt.subplot(1, 3, 3)
plt.plot(result['time'], result['UCAF'])
plt.xlabel('Time (s)')
plt.ylabel('UCAF')
plt.title('Air Flow')

plt.tight_layout()
plt.savefig('fmu_simulation_results.png', dpi=150)
plt.show()

print(f"\n✓ Simulación completada")
print(f"  UCAOT mean: {result['UCAOT'].mean():.2f} °C")
print(f"  UCWOT mean: {result['UCWOT'].mean():.2f} °C")
print(f"  UCAF mean: {result['UCAF'].mean():.2f}")
```

**Inspeccionar FMU:**
```python
from fmpy import dump

# Ver información del FMU
dump('deployment/fmu/HVACUnitCooler.fmu')
```

---

### 5. JModelica

```python
from pyfmi import load_fmu

# Cargar FMU
model = load_fmu('deployment/fmu/HVACUnitCooler.fmu')

# Configurar entradas
model.set('AMBT', 25.0)
model.set('UCTSP', 20.0)
model.set('CPSP', 10.0)
# ... (resto de inputs)

# Simular
res = model.simulate(start_time=0.0, final_time=100.0)

# Obtener resultados
ucaot = res['UCAOT']
ucwot = res['UCWOT']
ucaf = res['UCAF']
```

---

## 📊 Validación del FMU

**Verificar estructura:**
```bash
unzip -l HVACUnitCooler.fmu
```

**Contenido esperado:**
```
HVACUnitCooler.fmu
├── modelDescription.xml          (FMI 2.0 interface)
├── binaries/
│   ├── linux64/
│   │   └── HVACUnitCooler.so    (346 KB)
│   └── win64/
│       └── HVACUnitCooler.dll   (256 KB)
└── resources/
    ├── HVACUnitCooler_FMI2.py
    ├── pythonfmu/                (FMI 2.0 Python runtime)
    └── resources/
        ├── lightgbm_model.pkl   (2.2 MB - trained models)
        ├── scaler.pkl           (3.4 KB - StandardScaler)
        └── metadata.json        (1 KB - feature names)
```

---

## ⚡ Performance

| Métrica | Valor |
|---------|-------|
| **Latency (inferencia)** | < 1 ms |
| **Throughput** | > 1,000 predicciones/seg |
| **Precisión** | R²=0.993-1.0, MAPE=0.008-8.7% |
| **Tamaño FMU** | 2.9 MB |
| **Plataformas** | Linux x64, Windows x64 |

---

## 🔧 Troubleshooting

### Error: "Cannot load shared library"

**Linux:**
```bash
# Verificar permisos
chmod +x deployment/fmu/HVACUnitCooler.fmu

# Verificar dependencias
ldd binaries/linux64/HVACUnitCooler.so
```

**Windows:**
```cmd
# Verificar que Python esté instalado
python --version

# Instalar Visual C++ Redistributable si es necesario
```

### Error: "Python module not found"

El FMU incluye pythonfmu embebido, pero necesita Python runtime instalado:

```bash
# Linux/macOS
python3 --version  # Debe ser >= 3.7

# Windows
python --version   # Debe ser >= 3.7
```

### Error: "Model variables not initialized"

Asegurarse de proporcionar todas las 52 entradas con valores válidos. Valores por defecto en `HVACUnitCooler_FMI2.py` líneas 49-117.

---

## 📖 Especificaciones Técnicas

**FMI Version:** 2.0 Co-Simulation
**Model Identifier:** hvac_unit_cooler
**GUID:** 12345678-1234-5678-1234-567812345678
**Tool:** pythonfmu 0.6.2
**Generation Date:** 2025-11-20

**Inputs:** 52 variables (Real, continuous, input)
**Outputs:** 3 variables (Real, continuous, output, calculated)

**Communication Step Size:** Variable (canHandleVariableCommunicationStepSize=true)
**State:** Stateless (canGetAndSetFMUstate=false)

---

## 🎓 Ejemplo Completo: Co-Simulación

```python
#!/usr/bin/env python3
"""
Ejemplo completo de co-simulación con HVACUnitCooler.fmu
"""

from fmpy import simulate_fmu
import numpy as np
import matplotlib.pyplot as plt

# Simular 24 horas con step de 1 minuto
time_hours = 24
step_minutes = 1
n_steps = time_hours * 60 // step_minutes

# Crear perfil de temperatura ambiente (ciclo diario)
time = np.linspace(0, time_hours * 3600, n_steps)  # segundos
ambient_temp = 20 + 5 * np.sin(2 * np.pi * time / (24 * 3600))  # 20°C ± 5°C

# Crear perfil de setpoint (control)
setpoint = 18 * np.ones(n_steps)  # 18°C constante

# Inputs para FMU (valores típicos de operación)
inputs = {
    'AMBT': ambient_temp,
    'UCTSP': setpoint,
    'CPSP': 10.0,
    'UCAIT': 25.0,
    'CPPR': 2.0,
    'UCWF': 1.0,
    'CPMC': 50.0,
    'MVDP': 0.5,
    'CPCF': 1.5,
    'UCFS': 1500.0,
    'MVCV': 0.5,
    'UCHV': 0.3,
    'CPMV': 0.5,
    'UCHC': 0.2,
    'UCWIT': 15.0,
    'UCFMS': 1500.0,
    'CPDP': 0.3,
    'UCWDP': 0.2,
    'MVWF': 1.0,
    'UCOM': 1.0,
    # Engineered features (auto-calculadas en sistema real)
    'delta_T_water': 4.0,
    'delta_T_air': 2.5,
    'T_approach': 5.0,
    'T_water_avg': 13.0,
    'T_air_avg': 23.0,
    'mdot_water': 1000.0,
    'mdot_air': 1.2,
    'Q_water': 16.7,
    'Q_air': 15.0,
    'Q_avg': 15.85,
    'Q_imbalance': 1.7,
    'Q_imbalance_pct': 10.0,
    'efficiency_HX': 0.90,
    'effectiveness': 0.85,
    'NTU': 2.5,
    'C_ratio': 0.8,
    'Re_air_estimate': 5000.0,
    'flow_ratio': 1.2,
    'delta_T_ratio': 0.625,
    'setpoint_error': 3.0,
    'setpoint_error_abs': 3.0,
    'P_fan_estimate': 0.5,
    'P_pump_estimate': 0.3,
    'P_total_estimate': 0.8,
    'COP_estimate': 19.8,
    'time_index': 0.0,
    'cycle_hour': 0.5,
    'hour_sin': 0.0,
    'hour_cos': 1.0,
    'T_water_x_flow': 13.0,
    'T_air_x_flow': 27.6,
    'ambient_x_inlet': 625.0,
}

# Simular FMU
print("Iniciando co-simulación de 24 horas...")
result = simulate_fmu(
    'deployment/fmu/HVACUnitCooler.fmu',
    start_time=0.0,
    stop_time=time_hours * 3600,
    step_size=step_minutes * 60,
    start_values=inputs,
    output=['UCAOT', 'UCWOT', 'UCAF']
)

# Plotear resultados
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Temperatura de salida de aire
axes[0, 0].plot(result['time'] / 3600, result['UCAOT'], label='UCAOT (predicción)', linewidth=2)
axes[0, 0].plot(time / 3600, ambient_temp, '--', label='Temperatura ambiente', alpha=0.7)
axes[0, 0].axhline(y=setpoint[0], color='r', linestyle=':', label='Setpoint')
axes[0, 0].set_xlabel('Time (hours)')
axes[0, 0].set_ylabel('Temperature (°C)')
axes[0, 0].set_title('Unit Cooler Air Outlet Temperature')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Temperatura de salida de agua
axes[0, 1].plot(result['time'] / 3600, result['UCWOT'], label='UCWOT (predicción)', linewidth=2, color='blue')
axes[0, 1].set_xlabel('Time (hours)')
axes[0, 1].set_ylabel('Temperature (°C)')
axes[0, 1].set_title('Unit Cooler Water Outlet Temperature')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Flujo de aire
axes[1, 0].plot(result['time'] / 3600, result['UCAF'], label='UCAF (predicción)', linewidth=2, color='green')
axes[1, 0].set_xlabel('Time (hours)')
axes[1, 0].set_ylabel('Air Flow')
axes[1, 0].set_title('Unit Cooler Air Flow')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Estadísticas
stats_text = f"""
Estadísticas de Simulación (24h)

UCAOT (Air Outlet Temp):
  Mean:    {result['UCAOT'].mean():.2f} °C
  Std:     {result['UCAOT'].std():.2f} °C
  Min:     {result['UCAOT'].min():.2f} °C
  Max:     {result['UCAOT'].max():.2f} °C

UCWOT (Water Outlet Temp):
  Mean:    {result['UCWOT'].mean():.2f} °C
  Std:     {result['UCWOT'].std():.2f} °C
  Min:     {result['UCWOT'].min():.2f} °C
  Max:     {result['UCWOT'].max():.2f} °C

UCAF (Air Flow):
  Mean:    {result['UCAF'].mean():.2f}
  Std:     {result['UCAF'].std():.2f}
  Min:     {result['UCAF'].min():.2f}
  Max:     {result['UCAF'].max():.2f}

Simulación: {n_steps} steps, {step_minutes} min/step
"""
axes[1, 1].text(0.1, 0.5, stats_text, fontsize=9, family='monospace', verticalalignment='center')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('hvac_fmu_cosimulation_24h.png', dpi=150)
print("✓ Resultados guardados en hvac_fmu_cosimulation_24h.png")

plt.show()
```

---

## 📚 Referencias

- **FMI Standard 2.0:** https://fmi-standard.org/docs/2.0.3/
- **PythonFMU:** https://github.com/NTNU-IHB/PythonFMU
- **FMPy:** https://github.com/CATIA-Systems/FMPy
- **OpenModelica:** https://www.openmodelica.org/
- **Dymola:** https://www.3ds.com/products-services/catia/products/dymola/

---

## ✅ Checklist de Uso

- [ ] Descargar `HVACUnitCooler.fmu` (2.9 MB)
- [ ] Verificar Python >= 3.7 instalado
- [ ] Instalar herramienta de simulación (OpenModelica, MATLAB, FMPy, etc.)
- [ ] Importar FMU en herramienta
- [ ] Configurar 52 entradas con valores válidos
- [ ] Conectar 3 salidas (UCAOT, UCWOT, UCAF)
- [ ] Ejecutar simulación
- [ ] Validar resultados con R²=0.993-1.0

---

**Generado:** 2025-11-20
**Versión FMU:** 1.0.0
**Autor:** HVAC Digital Twin Team
**Contacto:** Ver README.md principal del proyecto

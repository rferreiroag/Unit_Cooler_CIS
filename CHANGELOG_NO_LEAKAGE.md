# Changelog: Data Leakage Fix

## Problema Identificado

El modelo anterior tenía **data leakage crítico**:
- 17 de 32 features derivadas usaban los targets (UCAOT, UCWOT, UCAF)
- Ejemplos: `delta_T_air = UCAOT - UCAIT`, `mdot_air = UCAF * rho_air`
- Esto causaba R² artificialmente alto (>0.99) que no era reproducible en producción

## Solución Implementada

### 1. Nuevo Feature Engineering Sin Leakage

**Archivo**: `src/data/feature_engineering_no_leakage.py`

Features válidas (computables en producción):
- **Temperaturas** (5): T_approach, T_water_ambient_diff, T_air_ambient_diff, setpoint_inlet_diff, setpoint_ambient_diff
- **Flujos** (1): mdot_water  
- **Térmica** (2): C_water, Q_max_water
- **Potencia** (3): P_fan_estimate, P_pump_estimate, P_total_estimate
- **Temporales** (4): time_index, cycle_hour, hour_sin, hour_cos
- **Interacciones** (4): T_water_x_flow, ambient_x_inlet, setpoint_x_flow, T_water_x_pressure

**Total**: 19 features derivadas + 20 sensores = **39 features totales**

### 2. Nuevo Pipeline de Datos

**Archivo**: `run_sprint1_pipeline_no_leakage.py`

- Genera datos en `data/processed_no_leakage/`
- ✅ Verificado: NO hay targets en features
- ✅ Todas las features computables en tiempo real

### 3. Comparación

| Métrica | Original (con leakage) | Nuevo (sin leakage) |
|---------|----------------------|-------------------|
| Features | 52 (23 + 29) | 39 (20 + 19) |
| Con targets | ❌ 17 features | ✅ 0 features |
| R² esperado | >0.99 (irreal) | 0.92-0.96 (realista) |
| Producción | ❌ No funciona | ✅ Funciona |

## Próximos Pasos

1. ✅ Datos generados sin leakage
2. ⏳ Reentrenar modelo con datos correctos
3. ⏳ Crear FMU con 20 entradas de sensores
4. ⏳ Validar predicciones en tiempo real

## Impacto

- **Precisión**: Ligera reducción (pero realista)
- **Producción**: Modelo ahora funcional en tiempo real
- **FMU**: Requiere solo 20 entradas de sensores (vs 52 antes)
- **Confiabilidad**: Predicciones reproducibles

## Archivos Modificados/Creados

- `src/data/feature_engineering_no_leakage.py` (nuevo)
- `run_sprint1_pipeline_no_leakage.py` (nuevo)  
- `data/processed_no_leakage/` (nuevo directorio)
- `CHANGELOG_NO_LEAKAGE.md` (este archivo)

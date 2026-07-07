#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Módulo de feature engineering para el modelado predictivo.

Genera variables explicativas a partir del dataset consolidado,incluyendo features de calendario, régimen geopolítico, lags temporales,medias móviles y contexto agregado nacional.

Proyecto: RepostaPro
Autor:    Víctor González Martín
"""

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ─── CONFIGURACIÓN ──────────────────────────────────────────────────────────

# Fecha de corte entre régimen pre-shock y post-shock.
# Justificación: primer día hábil tras el inicio del conflicto Irán-EEUU-Israel (28 feb 2026), precede al cruce efectivo de jerarquía detectado el 5 mar 2026.
FECHA_CORTE_SHOCK = pd.Timestamp("2026-03-01")

# Carburantes del núcleo predictivo (cobertura > 95% nacional)
CARBURANTES_NUCLEO = ["Precio Gasoleo A","Precio Gasolina 95 E5",]

# Lags temporales en días
LAGS = [1, 7, 30]

# Ventanas de las medias móviles en días
VENTANAS_MM = [7, 30]


# ─── FESTIVOS NACIONALES ESPAÑOLES ──────────────────────────────────────────
# Festivos fijos del calendario nacional. Festivos móviles (Semana Santa) se añaden manualmente para los años del dataset.

FESTIVOS_NACIONALES = {
    # Festivos fijos comunes a todos los años
    "01-01": "Año Nuevo",
    "01-06": "Reyes",
    "05-01": "Día del Trabajo",
    "08-15": "Asunción",
    "10-12": "Hispanidad",
    "11-01": "Todos los Santos",
    "12-06": "Constitución",
    "12-08": "Inmaculada",
    "12-25": "Navidad",}

# Festivos móviles (Semana Santa) por año
FESTIVOS_MOVILES = {
    2024: ["2024-03-28", "2024-03-29"],  # Jueves y Viernes Santo
    2025: ["2025-04-17", "2025-04-18"],
    2026: ["2026-04-02", "2026-04-03"],}


def es_festivo_nacional(fecha):
    """Determina si una fecha es festivo nacional español."""
    fecha_str = fecha.strftime("%m-%d")
    if fecha_str in FESTIVOS_NACIONALES:
        return True
    año = fecha.year
    fecha_iso = fecha.strftime("%Y-%m-%d")
    return fecha_iso in FESTIVOS_MOVILES.get(año, [])


# ─── FAMILIA 1: FEATURES DE CALENDARIO ──────────────────────────────────────

def generar_features_calendario(df):
    """Genera variables temporales derivadas de la fecha.

    Args:
        df: DataFrame con columna 'fecha' (datetime64).

    Returns:
        DataFrame con las nuevas columnas añadidas.
    """
    df = df.copy()
    df["dia_semana"] = df["fecha"].dt.dayofweek          # 0=lunes, 6=domingo
    df["mes"] = df["fecha"].dt.month                     # 1-12
    df["dia_del_mes"] = df["fecha"].dt.day               # 1-31
    df["semana_del_año"] = df["fecha"].dt.isocalendar().week.astype(int)
    df["es_fin_semana"] = df["dia_semana"].isin([5, 6])

    # Festivos: aplicar la función a cada fecha única (más eficiente)
    fechas_unicas = df["fecha"].drop_duplicates()
    mapa_festivos = {f: es_festivo_nacional(f) for f in fechas_unicas}
    df["es_festivo_nacional"] = df["fecha"].map(mapa_festivos)

    return df


# ─── FAMILIA 2: FEATURES DE RÉGIMEN ─────────────────────────────────────────

def generar_features_regimen(df):
    """Genera variables relacionadas con el shock geopolítico de marzo 2026.

    Args:
        df: DataFrame con columna 'fecha'.

    Returns:
        DataFrame con las columnas 'regimen' y 'dias_desde_shock'.
    """
    df = df.copy()

    df["regimen"] = np.where(df["fecha"] < FECHA_CORTE_SHOCK,
                              "pre_shock", "post_shock")
    df["dias_desde_shock"] = (df["fecha"] - FECHA_CORTE_SHOCK).dt.days

    return df


# ─── FAMILIA 3: FEATURES DE LAG ─────────────────────────────────────────────

def generar_features_lag(df, columna_precio, lags=LAGS):
    """Genera variables de lag temporal del precio, calculadas POR ESTACIÓN.

    Args:
        df: DataFrame con columnas 'IDEESS', 'fecha' y la columna de precio.
        columna_precio: nombre de la columna de precio.
        lags: lista de números de días de retardo.

    Returns:
        DataFrame con las nuevas columnas 'precio_lag_N' añadidas.
    """
    df = df.sort_values(["IDEESS", "fecha"]).copy()

    for lag in lags:
        nombre_col = f"precio_lag_{lag}"
        df[nombre_col] = df.groupby("IDEESS")[columna_precio].shift(lag)

    return df


# ─── FAMILIA 4: MEDIAS MÓVILES ──────────────────────────────────────────────

def generar_features_mm(df, columna_precio, ventanas=VENTANAS_MM):
    """Genera medias móviles del precio, calculadas POR ESTACIÓN.

    Args:
        df: DataFrame con columnas 'IDEESS', 'fecha' y la columna de precio.
        columna_precio: nombre de la columna de precio.
        ventanas: lista de tamaños de ventana en días.

    Returns:
        DataFrame con las nuevas columnas 'precio_mm_N' añadidas.
    """
    df = df.sort_values(["IDEESS", "fecha"]).reset_index(drop=True).copy()

    for ventana in ventanas:
        nombre_col = f"precio_mm_{ventana}"
        # Para evitar data leakage: la media móvil del día t NO incluye el día t.
        # Estrategia: calcular precio_shifted (shift de 1) y luego rolling sobre él.
        precio_shifted = df.groupby("IDEESS")[columna_precio].shift(1)
        df[nombre_col] = (precio_shifted
                          .groupby(df["IDEESS"])
                          .rolling(window=ventana, min_periods=1)
                          .mean()
                          .reset_index(level=0, drop=True))

    return df


# ─── FAMILIA 5: CONTEXTO AGREGADO NACIONAL ──────────────────────────────────

def generar_features_nacionales(df, columna_precio):
    """Genera variables de contexto agregado nacional para cada día.

    Args:
        df: DataFrame con columnas 'fecha' y la columna de precio.
        columna_precio: nombre de la columna de precio.

    Returns:
        DataFrame con las nuevas columnas añadidas.
    """
    df = df.copy()

    # Precio medio nacional por día
    precio_nacional_dia = (df.groupby("fecha")[columna_precio]
                            .transform("mean"))
    df["precio_medio_nacional_dia"] = precio_nacional_dia

    # Diferencial de cada estación respecto a la media nacional del día
    df["diferencial_vs_nacional"] = df[columna_precio] - precio_nacional_dia

    return df


# ─── FUNCIÓN PRINCIPAL: PREPARAR DATASET POR CARBURANTE ─────────────────────

def preparar_dataset_carburante(df_consolidado, columna_carburante):
    """Construye el dataset analítico final para un carburante concreto.

    Filtra las filas donde el carburante tiene precio (no NaN), aplica las
    cinco familias de features y devuelve el dataset listo para modelado.

    Args:
        df_consolidado: DataFrame con el histórico completo (los 3 Parquet unidos).
        columna_carburante: nombre de la columna del carburante a modelar.

    Returns:
        DataFrame con todas las features añadidas y filtrado al carburante objetivo.
    """
    print(f"Preparando dataset para: {columna_carburante}")
    print(f"  Filas de entrada:      {len(df_consolidado):,}")

    # Filtrar filas donde el carburante tiene precio declarado
    df = df_consolidado[df_consolidado[columna_carburante].notna()].copy()
    print(f"  Filas tras filtrado:   {len(df):,} (cobertura: {len(df)/len(df_consolidado)*100:.1f}%)")

    # Aplicar las cinco familias de features
    df = generar_features_calendario(df)
    print(f"  ✓ Familia 1 (calendario) aplicada")

    df = generar_features_regimen(df)
    print(f"  ✓ Familia 2 (régimen) aplicada")

    df = generar_features_lag(df, columna_carburante)
    print(f"  ✓ Familia 3 (lags) aplicada")

    df = generar_features_mm(df, columna_carburante)
    print(f"  ✓ Familia 4 (medias móviles) aplicada")

    df = generar_features_nacionales(df, columna_carburante)
    print(f"  ✓ Familia 5 (contexto nacional) aplicada")

    # Renombrar la columna de precio objetivo para uniformidad
    df = df.rename(columns={columna_carburante: "precio"})

    print(f"  Filas finales:         {len(df):,}")
    print(f"  Columnas finales:      {df.shape[1]}")

    return df


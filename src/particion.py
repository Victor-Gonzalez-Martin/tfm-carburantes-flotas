#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Módulo de partición temporal del dataset para modelado predictivo.

Aplica una estrategia de partición train / validation / test que respeta la cronología de la serie temporal, evitando data leakage entre conjuntos.

Proyecto: RepostaPro
Autor:    Víctor González Martín
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


# ─── CONFIGURACIÓN DE FECHAS DE CORTE ──────────────────────────────────────

# Fechas de corte para PRE-SHOCK (régimen normal, 787 días totales)
# Train: ene 2024 → sep 2025 (75%) , Validation: oct 2025 → dic 2025 (12,5%) y Test: ene 2026 → 28 feb 2026 (12,5%)
CORTE_PRE_TRAIN_VAL  = pd.Timestamp("2025-10-01")
CORTE_PRE_VAL_TEST   = pd.Timestamp("2026-01-01")
FIN_PRE_SHOCK        = pd.Timestamp("2026-02-28")

# Fechas de corte para POST-SHOCK (régimen nuevo, 106 días totales)
# Train: 1 mar 2026 → 30 abr 2026 (~58%), Validation: 1 may 2026 → 24 may 2026 (~22%) y Test: 25 may 2026 → 14 jun 2026 (~20%)
INICIO_POST_SHOCK    = pd.Timestamp("2026-03-01")
CORTE_POST_TRAIN_VAL = pd.Timestamp("2026-05-01")
CORTE_POST_VAL_TEST  = pd.Timestamp("2026-05-25")


def particionar_pre_shock(df):
    """Particiona el subconjunto pre-shock en train / validation / test.

    Args:
        df: DataFrame con la columna 'fecha' y todas las features generadas.

    Returns:
        dict con tres DataFrames: 'train', 'val', 'test'.
    """
    df_pre = df[df["fecha"] <= FIN_PRE_SHOCK].copy()

    train = df_pre[df_pre["fecha"] < CORTE_PRE_TRAIN_VAL].copy()
    val   = df_pre[(df_pre["fecha"] >= CORTE_PRE_TRAIN_VAL) &
                   (df_pre["fecha"] < CORTE_PRE_VAL_TEST)].copy()
    test  = df_pre[df_pre["fecha"] >= CORTE_PRE_VAL_TEST].copy()

    return {"train": train, "val": val, "test": test}


def particionar_post_shock(df):
    """Particiona el subconjunto post-shock en train / validation / test.

    Args:
        df: DataFrame con la columna 'fecha' y todas las features generadas.

    Returns:
        dict con tres DataFrames: 'train', 'val', 'test'.
    """
    df_post = df[df["fecha"] >= INICIO_POST_SHOCK].copy()

    train = df_post[df_post["fecha"] < CORTE_POST_TRAIN_VAL].copy()
    val   = df_post[(df_post["fecha"] >= CORTE_POST_TRAIN_VAL) &
                    (df_post["fecha"] < CORTE_POST_VAL_TEST)].copy()
    test  = df_post[df_post["fecha"] >= CORTE_POST_VAL_TEST].copy()

    return {"train": train, "val": val, "test": test}


def resumen_particion(particiones, nombre_regimen):
    """Imprime un resumen de la partición realizada.

    Args:
        particiones: dict con 'train', 'val', 'test'.
        nombre_regimen: string para etiquetar la salida.
    """
    print(f"\n{'='*70}")
    print(f"PARTICIÓN — Régimen {nombre_regimen}")
    print(f"{'='*70}")

    total = sum(len(p) for p in particiones.values())

    for nombre, df_p in particiones.items():
        if len(df_p) == 0:
            print(f"  {nombre.upper():<12} (vacío)")
            continue
        pct = len(df_p) / total * 100 if total > 0 else 0
        fecha_min = df_p["fecha"].min().date()
        fecha_max = df_p["fecha"].max().date()
        dias = df_p["fecha"].nunique()
        print(f"  {nombre.upper():<12} {len(df_p):>10,} filas ({pct:5.1f}%)  "
              f"{fecha_min} → {fecha_max}  ({dias} días)")


def guardar_particiones(particiones, nombre_carburante, regimen, carpeta_destino):
    """Guarda las particiones en formato Parquet.

    Args:
        particiones: dict con 'train', 'val', 'test'.
        nombre_carburante: string para el nombre del fichero (ej. 'gasoleo_a').
        regimen: string ('pre_shock' o 'post_shock').
        carpeta_destino: Path donde guardar los Parquet.
    """
    carpeta_destino.mkdir(parents=True, exist_ok=True)

    for nombre, df_p in particiones.items():
        if len(df_p) == 0:
            continue
        ruta = carpeta_destino / f"{nombre_carburante}_{regimen}_{nombre}.parquet"
        df_p.to_parquet(ruta, compression="snappy", index=False)


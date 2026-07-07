#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Módulo de consolidación de los ficheros JSON descargados a Parquet analítico.

Itera los JSON día a día, aplica limpiezas estándar (conversión de tipos, gestión de valores ausentes, normalización del campo Rótulo y exclusión de outliers geográficos) y los consolida en un único Parquet por año.

Proyecto: RepostaPro
Autor:    Víctor González Martín
"""

import json
import logging
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ─── CONFIGURACIÓN ──────────────────────────────────────────────────────────

# Campos del JSON que se conservan en el dataset analítico final.
# Se excluyen los carburantes con cobertura inferior al 5 % nacional para reducir el tamaño y el ruido del dataset.
COLUMNAS_IDENTIFICACION = ["IDEESS","IDMunicipio","IDProvincia","IDCCAA","C.P."]

COLUMNAS_UBICACION = ["Dirección","Localidad","Municipio","Provincia","Latitud","Longitud (WGS84)"]

COLUMNAS_DESCRIPTIVAS = ["Rótulo","Horario","Tipo Venta"]

COLUMNAS_PRECIOS = ["Precio Gasoleo A","Precio Gasoleo Premium","Precio Gasolina 95 E5","Precio Gasolina 98 E5","Precio Gases licuados del petróleo","Precio Adblue"]

COLUMNAS_COMPOSICION = ["% BioEtanol","% Éster metílico"]

COLUMNAS_TODAS = (COLUMNAS_IDENTIFICACION+ COLUMNAS_UBICACION+ COLUMNAS_DESCRIPTIVAS+ COLUMNAS_PRECIOS+ COLUMNAS_COMPOSICION)


# ─── DICCIONARIO DE NORMALIZACIÓN DE MARCAS ─────────────────────────────────
# Cada clave es la marca normalizada canónica;cada valor es la lista de patrones que, si aparecen como subcadena del rótulo,asignan la estación a esa marca. Idéntico al usado en el notebook 01.

GRUPOS_MARCA = {
    "CEPSA-MOEVE":  ["CEPSA", "MOEVE"],
    "REPSOL":       ["REPSOL"],
    "PETRONOR":     ["PETRONOR"],
    "GALP":         ["GALP"],
    "BALLENOIL":    ["BALLENOIL"],
    "PLENERGY":     ["PLENERGY"],
    "SHELL":        ["SHELL"],
    "PETROPRIX":    ["PETROPRIX"],
    "CARREFOUR":    ["CARREFOUR"],
    "BP":           ["BP "],
    "AVIA":         ["AVIA"],
    "Q8":           ["Q8"],
    "BONAREA":      ["BONAREA", "BONÀREA"],
    "ESCLATOIL":    ["ESCLATOIL"],
    "ALCAMPO":      ["ALCAMPO"],
    "ENI":          ["ENI "],
    "CAMPSA":       ["CAMPSA"],
    "VALCARCE":     ["VALCARCE"],
    "AGLA":         ["AGLA"],
    "DISA":         ["DISA"],
}

# Falsos positivos detectados durante la validación del Sprint 1.
FALSOS_POSITIVOS = ["AN ENERGETICOS - MENDAVIA"]

# Bounding box del territorio español para detectar outliers geográficos.
LAT_MIN_ESPANA = 27.0
LAT_MAX_ESPANA = 44.0
LON_MIN_ESPANA = -18.5
LON_MAX_ESPANA = 5.0

# ─── FUNCIONES DE LIMPIEZA ──────────────────────────────────────────────────

def normalizar_rotulo(rotulo_original):
    """Devuelve la marca normalizada según el diccionario GRUPOS_MARCA.

    Si el rótulo no encaja con ningún patrón, se devuelve sin cambios.
    Los falsos positivos detectados durante el Sprint 1 se preservan tal cual.
    """
    if pd.isna(rotulo_original):
        return rotulo_original

    rotulo_limpio = str(rotulo_original).upper().strip()

    # Salvaguarda contra falsos positivos conocidos
    if rotulo_original in FALSOS_POSITIVOS:
        return rotulo_original

    for marca_normalizada, patrones in GRUPOS_MARCA.items():
        for patron in patrones:
            if patron in rotulo_limpio:
                return marca_normalizada

    return rotulo_original


def cargar_y_limpiar_json(ruta_json, fecha):
    """Carga un JSON de la API, extrae las estaciones y aplica la limpieza estándar.

    Args:
        ruta_json: Path al fichero JSON descargado.
        fecha: objeto date correspondiente a esta consulta.

    Returns:
        DataFrame con las estaciones de esa fecha, ya limpio y normalizado.
        Devuelve None si el fichero está corrupto o vacío.
    """
    try:
        with open(ruta_json, "r", encoding="utf-8") as f:
            datos = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    estaciones = datos.get("ListaEESSPrecio", [])
    if not estaciones:
        return None

    # Convertir lista de diccionarios a DataFrame
    df = pd.json_normalize(estaciones)

    # Quedarnos solo con las columnas que necesitamos
    columnas_presentes = [c for c in COLUMNAS_TODAS if c in df.columns]
    df = df[columnas_presentes].copy()

    # 1) Cadenas vacías → NaN
    df = df.replace("", np.nan)

    # 2) Conversión de comas decimales a punto y a float
    columnas_numericas = ([c for c in COLUMNAS_PRECIOS if c in df.columns]+ [c for c in ["Latitud", "Longitud (WGS84)"] if c in df.columns]+ [c for c in COLUMNAS_COMPOSICION if c in df.columns])
    for col in columnas_numericas:
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3) Normalización del rótulo
    if "Rótulo" in df.columns:
        df["Rotulo_normalizado"] = df["Rótulo"].apply(normalizar_rotulo)

    # 4) Marcar outliers geográficos (no se eliminan, se etiquetan)
    if {"Latitud", "Longitud (WGS84)"}.issubset(df.columns):
        df["outlier_geografico"] = ~(df["Latitud"].between(LAT_MIN_ESPANA, LAT_MAX_ESPANA)& df["Longitud (WGS84)"].between(LON_MIN_ESPANA, LON_MAX_ESPANA))
    else:
        df["outlier_geografico"] = False

    # 5) Añadir la fecha del snapshot como nueva columna
    df["fecha"] = pd.Timestamp(fecha)

    return df


# ─── FUNCIÓN PRINCIPAL DE CONSOLIDACIÓN ─────────────────────────────────────

def parsear_fecha_de_nombre(nombre_fichero):
    """Extrae la fecha del nombre de un fichero con formato precios_AAAA-MM-DD.json"""
    coincidencia = re.match(r"precios_(\d{4})-(\d{2})-(\d{2})\.json", nombre_fichero)
    if not coincidencia:
        return None
    ano, mes, dia = map(int, coincidencia.groups())
    return date(ano, mes, dia)


def consolidar_ano(ano, carpeta_raw, carpeta_processed, logger=None):
    """Consolida los JSON de un año concreto en un único Parquet.

    Args:
        ano: año a consolidar (int).
        carpeta_raw: Path con los JSON descargados.
        carpeta_processed: Path donde guardar el Parquet resultante.
        logger: logger opcional para registrar el progreso.

    Returns:
        dict con resumen: ano, n_ficheros, n_filas, n_columnas, tamano_mb, ruta_salida.
    """
    if logger is None:
        logger = logging.getLogger("consolidacion")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",datefmt="%Y-%m-%d %H:%M:%S"))
            logger.addHandler(ch)

    carpeta_processed.mkdir(parents=True, exist_ok=True)

    # Identificar los ficheros JSON del año
    todos_los_json = sorted(carpeta_raw.glob("precios_*.json"))
    json_del_ano = []
    for ruta in todos_los_json:
        fecha = parsear_fecha_de_nombre(ruta.name)
        if fecha is not None and fecha.year == ano:
            json_del_ano.append((fecha, ruta))

    if not json_del_ano:
        logger.warning("No se encontraron ficheros JSON para el año %s", ano)
        return None

    logger.info("=" * 70)
    logger.info("CONSOLIDANDO AÑO %s", ano)
    logger.info("Ficheros a procesar: %s", len(json_del_ano))
    logger.info("=" * 70)
    inicio = datetime.now()
    dataframes = []
    omitidos = 0

    for i, (fecha, ruta) in enumerate(json_del_ano, 1):
        df_dia = cargar_y_limpiar_json(ruta, fecha)
        if df_dia is None:
            logger.warning("  [%s/%s] OMITIDO (fichero corrupto): %s",i, len(json_del_ano), fecha)
            omitidos += 1
            continue

        dataframes.append(df_dia)

        if i % 30 == 0 or i == len(json_del_ano):
            logger.info("  [%s/%s] Procesados hasta %s", i, len(json_del_ano), fecha)

    # Concatenar todos los DataFrames del año
    df_anual = pd.concat(dataframes, ignore_index=True)

    # Guardar como Parquet con compresión snappy (rápido y eficiente)
    ruta_salida = carpeta_processed / f"historico_carburantes_{ano}.parquet"
    df_anual.to_parquet(ruta_salida, compression="snappy", index=False)

    duracion = datetime.now() - inicio
    tamano_mb = ruta_salida.stat().st_size / 1024 / 1024

    logger.info("=" * 70)
    logger.info("AÑO %s CONSOLIDADO", ano)
    logger.info("Filas:      %s", f"{len(df_anual):,}")
    logger.info("Columnas:   %s", df_anual.shape[1])
    logger.info("Omitidos:   %s ficheros", omitidos)
    logger.info("Duración:   %s", duracion)
    logger.info("Tamaño:     %.2f MB", tamano_mb)
    logger.info("Destino:    %s", ruta_salida)
    logger.info("=" * 70)

    return {
        "ano": ano,
        "n_ficheros": len(json_del_ano),
        "n_filas": len(df_anual),
        "n_columnas": df_anual.shape[1],
        "tamano_mb": tamano_mb,
        "ruta_salida": ruta_salida,
        "omitidos": omitidos,
        "duracion": duracion}


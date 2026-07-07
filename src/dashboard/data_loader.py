"""
Funciones de carga de datos compartidas entre las pestañas del dashboard.
Usa @st.cache_data para no recargar en cada interacción del usuario.
"""

from pathlib import Path
import logging

import pandas as pd
import streamlit as st


# Configuración de rutas (relativas a la raíz del proyecto)
RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
CARPETA_RESULTADOS = RAIZ_PROYECTO / "outputs" / "resultados_modelos"
CARPETA_MODELOS_LGB = RAIZ_PROYECTO / "outputs" / "models" / "lightgbm"
CARPETA_VALIDACION = RAIZ_PROYECTO / "data" / "validacion"
CARPETA_FIGURAS = RAIZ_PROYECTO / "outputs" / "figures"


@st.cache_data
def cargar_predicciones_validacion():
    """Carga el dataset de predicciones del modelo de producción (Sprint 4 Fase 1).
    
    Cacheado: solo se carga la primera vez. Las siguientes llamadas son instantáneas.
    """
    ruta = CARPETA_RESULTADOS / "predicciones_validacion_modelo_produccion.parquet"
    df = pd.read_parquet(ruta)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


@st.cache_data
def cargar_resultados_modelos():
    """Carga los CSV con métricas de los 4 modelos (baseline + Prophet + LightGBM)."""
    resultados = {}
    
    ruta_baseline = CARPETA_RESULTADOS / "baseline_resultados.csv"
    ruta_baseline_nac = CARPETA_RESULTADOS / "baseline_nacional_resultados.csv"
    ruta_prophet = CARPETA_RESULTADOS / "prophet_resultados.csv"
    ruta_lgb = CARPETA_RESULTADOS / "lightgbm_resultados.csv"
    ruta_comparativa = CARPETA_RESULTADOS / "comparativa_modelos_test.csv"
    ruta_validacion = CARPETA_RESULTADOS / "validacion_walking_forward.csv"
    ruta_economico = CARPETA_RESULTADOS / "analisis_economico_3_flotas.csv"
    
    for nombre, ruta in [
        ("baseline", ruta_baseline),
        ("baseline_nacional", ruta_baseline_nac),
        ("prophet", ruta_prophet),
        ("lightgbm", ruta_lgb),
        ("comparativa", ruta_comparativa),
        ("validacion", ruta_validacion),
        ("economico", ruta_economico),
    ]:
        if ruta.exists():
            resultados[nombre] = pd.read_csv(ruta)
    
    return resultados


@st.cache_resource
def cargar_modelo_lightgbm(carburante_key, regimen):
    """Carga un modelo LightGBM serializado.
    
    Usa @st.cache_resource (no @st.cache_data) porque los modelos no son DataFrames.
    
    Args:
        carburante_key: 'gasoleo_a' o 'gasolina_95'.
        regimen: 'pre_shock' o 'post_shock'.
    
    Returns:
        Booster LightGBM cargado.
    """
    import lightgbm as lgb
    ruta = CARPETA_MODELOS_LGB / f"lightgbm_{carburante_key}_{regimen}.txt"
    return lgb.Booster(model_file=str(ruta))


@st.cache_data
def obtener_fechas_disponibles():
    """Devuelve la lista de fechas disponibles en el dataset de predicciones."""
    df = cargar_predicciones_validacion()
    return sorted(df["fecha"].dt.date.unique())


@st.cache_data
def obtener_carburantes_disponibles():
    """Devuelve la lista de carburantes disponibles."""
    df = cargar_predicciones_validacion()
    return sorted(df["carburante"].unique().tolist())


@st.cache_data
def obtener_provincias_disponibles():
    """Devuelve la lista de provincias disponibles para filtros."""
    df = cargar_predicciones_validacion()
    return sorted(df["provincia"].dropna().unique().tolist()) 

# ─── ACTUALIZACIÓN DE DATOS DESDE EL MINISTERIO ─────────────────────────────

def actualizar_datos_desde_ministerio(progress_callback=None, dias_horizonte_futuro=2):
    """Descarga datos hasta HOY (tiempo real) y genera predicciones recursivas hacia el futuro.
    
    ARQUITECTURA DE DESCARGA HÍBRIDA:
    - Endpoint HISTÓRICO para días anteriores faltantes (con 1 día latencia).
    - Endpoint EN VIVO para el día actual (datos actualizados al momento).
    
    Flujo:
    1. Detecta el último día disponible en disco.
    2. Descarga días faltantes anteriores con endpoint histórico.
    3. Descarga el día actual con endpoint en vivo.
    4. Procesa todos los JSONs.
    5. Genera predicciones reales (días con datos) + recursivas (HOY+1, HOY+2).
    6. Guarda Parquet de predicciones.
    
    Args:
        progress_callback: función opcional para reportar progreso.
        dias_horizonte_futuro: días futuros a predecir.
    
    Returns:
        Dict con resumen + marca de actualización del Ministerio.
    """
    import logging
    from datetime import date, timedelta
    
    import sys
    sys.path.insert(0, str(RAIZ_PROYECTO))
    
    from src.descarga import descargar_rango
    from src.consolidacion import cargar_y_limpiar_json
    from src.features import preparar_dataset_carburante
    from src.modelos import preparar_datos_lightgbm, predecir_recursivo_multi_dia
    from src.dashboard.descarga_tiempo_real import descargar_tiempo_real
    
    hoy = date.today()
    ayer = hoy - timedelta(days=1)
    
    # 1. Detectar último día disponible en disco
    jsons_disponibles = sorted(CARPETA_VALIDACION.glob("precios_*.json"))
    fechas_en_disco = []
    for ruta_json in jsons_disponibles:
        fecha_str = ruta_json.stem.replace("precios_", "")
        try:
            fecha_str_clean = fecha_str.replace("-", "")
            if len(fecha_str_clean) == 8:
                fecha = pd.Timestamp(f"{fecha_str_clean[:4]}-{fecha_str_clean[4:6]}-{fecha_str_clean[6:8]}")
                fechas_en_disco.append(fecha.date())
        except Exception:
            continue
    
    ultimo_dia_en_disco = max(fechas_en_disco) if fechas_en_disco else date(2026, 6, 14)
    
    # 2. Descargar días anteriores que falten (endpoint histórico)
    logger = logging.getLogger("actualizar_dashboard")
    
    if ultimo_dia_en_disco < ayer:
        if progress_callback:
            progress_callback("Descargando días anteriores (histórico)...", 0.1)
        
        fecha_inicio_hist = ultimo_dia_en_disco + timedelta(days=1)
        ruta_log = RAIZ_PROYECTO / "logs" / f"descarga_dashboard_{hoy.isoformat()}.log"
        ruta_log.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            descargar_rango(
                fecha_inicio=fecha_inicio_hist,
                fecha_fin=ayer,
                carpeta_raw=CARPETA_VALIDACION,
                ruta_log=ruta_log,
            )
        except Exception as e:
            return {
                "estado": "error",
                "ultimo_dia": ultimo_dia_en_disco,
                "mensaje": f"Error descargando histórico: {str(e)}",
                "dias_descargados": 0,
            }
    
    # 3. Intentar descargar el día actual (endpoint en vivo).
    # Si falla, seguimos con los datos descargados del histórico.
    if progress_callback:
        progress_callback("Intentando descargar datos en tiempo real...", 0.3)
    
    ruta_hoy = CARPETA_VALIDACION / f"precios_{hoy.isoformat()}.json"
    
    # Solo intentar el endpoint en vivo si NO tenemos ya el día actual en disco
    if not ruta_hoy.exists() or ruta_hoy.stat().st_size < 1024 * 1024:
        try:
            resultado_vivo = descargar_tiempo_real(ruta_hoy, logger=logger)
        except Exception as e:
            logger.warning(f"Excepción en descarga tiempo real: {e}")
            resultado_vivo = {"exito": False}
    else:
        resultado_vivo = {
            "exito": True,
            "fecha_actualizacion": "ya descargado",
            "n_estaciones": 0,}
    
    fecha_actualizacion_ministerio = resultado_vivo.get("fecha_actualizacion", "desconocida")
    descarga_vivo_exitosa = resultado_vivo.get("exito", False)
    
    if not descarga_vivo_exitosa:
        logger.warning("No se pudieron descargar datos en vivo. Continuando con datos del histórico.")
    
    # 4. Consolidar TODOS los JSONs disponibles
    if progress_callback:
        progress_callback("Consolidando datos...", 0.5)
    
    df_historico = pd.read_parquet(
        RAIZ_PROYECTO / "data" / "processed" / "historico_carburantes_2026.parquet"
    )
    
    dataframes_validacion = []
    for ruta_json in sorted(CARPETA_VALIDACION.glob("precios_*.json")):
        fecha_str = ruta_json.stem.replace("precios_", "")
        try:
            fecha_str_clean = fecha_str.replace("-", "")
            if len(fecha_str_clean) != 8:
                continue
            fecha = pd.Timestamp(f"{fecha_str_clean[:4]}-{fecha_str_clean[4:6]}-{fecha_str_clean[6:8]}")
            df_dia = cargar_y_limpiar_json(ruta_json, fecha)
            dataframes_validacion.append(df_dia)
        except Exception:
            continue
    
    if not dataframes_validacion:
        return {
            "estado": "error",
            "ultimo_dia": ultimo_dia_en_disco,
            "mensaje": "No se pudieron cargar JSONs.",
            "dias_descargados": 0,
        }
    
    df_validacion = pd.concat(dataframes_validacion, ignore_index=True)
    
    columnas_comunes = list(set(df_historico.columns) & set(df_validacion.columns))
    df_combinado = pd.concat([
        df_historico[columnas_comunes],
        df_validacion[columnas_comunes],
    ], ignore_index=True)
    df_combinado = df_combinado.drop_duplicates(subset=["IDEESS", "fecha"]).sort_values(
        ["IDEESS", "fecha"]
    ).reset_index(drop=True)
    
    ultimo_dia_disponible = df_validacion["fecha"].max().date()
    
    # 5. Generar predicciones reales + recursivas
    if progress_callback:
        progress_callback("Generando predicciones LightGBM...", 0.7)
    
    todas_fechas_reales = sorted(df_validacion["fecha"].unique())
    fechas_futuras = [
        pd.Timestamp(ultimo_dia_disponible) + pd.Timedelta(days=d)
        for d in range(1, dias_horizonte_futuro + 1)
    ]
    
    predicciones_acumuladas = []
    
    for carburante_key, nombre_carburante, col_precio in [
        ("gasoleo_a", "Gasóleo A", "Precio Gasoleo A"),
        ("gasolina_95", "Gasolina 95 E5", "Precio Gasolina 95 E5"),
    ]:
        modelo = cargar_modelo_lightgbm(carburante_key, "pre_shock")
        
        # 5a. Predicciones para días reales (1 paso)
        df_features = preparar_dataset_carburante(df_combinado, col_precio)
        df_eval_real = df_features[df_features["fecha"].isin(todas_fechas_reales)].copy()
        
        X, _ = preparar_datos_lightgbm(df_eval_real)
        y_pred = modelo.predict(X)
        
        df_real = df_eval_real[["IDEESS", "fecha", "Rotulo_normalizado", "Provincia",
                                  "precio", "Latitud", "Longitud (WGS84)"]].copy()
        df_real.columns = ["IDEESS", "fecha", "marca", "provincia",
                            "precio_real", "latitud", "longitud"]
        df_real["precio_predicho"] = y_pred
        df_real["error_abs"] = (df_real["precio_real"] - df_real["precio_predicho"]).abs()
        df_real["carburante"] = nombre_carburante
        df_real["tipo_prediccion"] = "real"
        df_real["dia_horizonte"] = 0
        predicciones_acumuladas.append(df_real)
        
        # 5b. Predicciones recursivas para días futuros
        if progress_callback:
            progress_callback(f"Predicciones recursivas ({nombre_carburante})...", 0.85)
        
        df_pred_futuras = predecir_recursivo_multi_dia(
            modelo=modelo,
            df_combinado=df_combinado,
            fechas_futuras=fechas_futuras,
            columna_precio=col_precio,
        )
        
        if not df_pred_futuras.empty:
            df_pred_futuras["precio_real"] = pd.NA
            df_pred_futuras["error_abs"] = pd.NA
            df_pred_futuras["carburante"] = nombre_carburante
            df_pred_futuras["tipo_prediccion"] = "futura"
            predicciones_acumuladas.append(df_pred_futuras)
    
    df_predicciones_final = pd.concat(predicciones_acumuladas, ignore_index=True)
    
    # 6. Guardar
    if progress_callback:
        progress_callback("Guardando dataset...", 0.95)
    
    ruta_predicciones = CARPETA_RESULTADOS / "predicciones_validacion_modelo_produccion.parquet"
    df_predicciones_final.to_parquet(ruta_predicciones, compression="snappy", index=False)
    
    # 7. Limpiar caché
    cargar_predicciones_validacion.clear()
    obtener_fechas_disponibles.clear()
    
    if progress_callback:
        progress_callback("✓ Actualización completa", 1.0)
    
    return {
        "estado": "actualizado",
        "ultimo_dia": ultimo_dia_disponible,
        "ultimo_dia_predicho": fechas_futuras[-1].date() if fechas_futuras else ultimo_dia_disponible,
        "mensaje": (
            f"Datos actualizados hasta {ultimo_dia_disponible}. "
            f"Predicciones generadas para {len(fechas_futuras)} días futuros."
        ),
        "fecha_actualizacion_ministerio": fecha_actualizacion_ministerio,
        "descarga_vivo_exitosa": descarga_vivo_exitosa,
        "n_estaciones_vivo": resultado_vivo.get("n_estaciones", 0) if descarga_vivo_exitosa else 0,
    }
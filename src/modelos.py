#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Módulo común de utilidades de modelado predictivo.

Proporciona funciones reutilizables para los tres tipos de modelos del proyecto:
- Métricas de evaluación (MAE, RMSE, MAPE)
- Evaluación común train/val/test
- Guardado y carga de modelos serializados

Proyecto: RepostaPro
Autor:    Víctor González Martín
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ─── MÉTRICAS DE EVALUACIÓN ─────────────────────────────────────────────────

def mae(y_real, y_predicho):
    """Mean Absolute Error - error medio en valor absoluto.

    Interpretable directamente como céntimos por litro de error medio.
    """
    return np.mean(np.abs(y_real - y_predicho))


def rmse(y_real, y_predicho):
    """Root Mean Squared Error - raíz del error cuadrático medio.

    Penaliza más los errores grandes que MAE.
    """
    return np.sqrt(np.mean((y_real - y_predicho) ** 2))


def mape(y_real, y_predicho):
    """Mean Absolute Percentage Error - porcentaje de error medio.

    Útil para comparar errores entre productos de distinto nivel de precio.
    """
    # Evitar división por cero o por valores nulos
    mascara = y_real != 0
    return np.mean(np.abs((y_real[mascara] - y_predicho[mascara]) / y_real[mascara])) * 100


def evaluar(y_real, y_predicho, nombre=""):
    """Calcula las tres métricas y devuelve un dict.

    Excluye automáticamente filas con NaN en cualquiera de los dos arrays.
    """
    y_real = pd.Series(y_real).reset_index(drop=True)
    y_predicho = pd.Series(y_predicho).reset_index(drop=True)

    # Excluir NaN
    mascara = (~y_real.isna()) & (~y_predicho.isna())
    y_real_clean = y_real[mascara].values
    y_predicho_clean = y_predicho[mascara].values

    if len(y_real_clean) == 0:
        return {
            "nombre": nombre,
            "n_observaciones": 0,
            "mae_cts": np.nan,
            "rmse_cts": np.nan,
            "mape_pct": np.nan,}

    return {
        "nombre": nombre,
        "n_observaciones": len(y_real_clean),
        "mae_cts": mae(y_real_clean, y_predicho_clean) * 100,  # en céntimos
        "rmse_cts": rmse(y_real_clean, y_predicho_clean) * 100,
        "mape_pct": mape(y_real_clean, y_predicho_clean),}


def imprimir_evaluacion(resultados):
    """Imprime resultados de evaluación en tabla legible.

    Args:
        resultados: lista de dicts devueltos por evaluar().
    """
    print(f"{'Conjunto':<35} {'N obs':>12} {'MAE (cts)':>12} "
          f"{'RMSE (cts)':>12} {'MAPE (%)':>12}")
    print("-" * 90)
    for r in resultados:
        n_obs = f"{r['n_observaciones']:,}"
        if np.isnan(r['mae_cts']):
            print(f"{r['nombre']:<35} {n_obs:>12} {'—':>12} {'—':>12} {'—':>12}")
        else:
            print(f"{r['nombre']:<35} {n_obs:>12} "
                  f"{r['mae_cts']:>11.3f}  "
                  f"{r['rmse_cts']:>11.3f}  "
                  f"{r['mape_pct']:>11.3f}")


# ─── CARGA DE PARTICIONES ───────────────────────────────────────────────────

def cargar_particion(carpeta_particiones, carburante, regimen, conjunto):
    """Carga una partición específica desde disco.

    Args:
        carpeta_particiones: Path con los Parquet de particiones.
        carburante: 'gasoleo_a' o 'gasolina_95'.
        regimen: 'pre_shock' o 'post_shock'.
        conjunto: 'train', 'val' o 'test'.

    Returns:
        DataFrame con la partición solicitada.
    """
    nombre = f"{carburante}_{regimen}_{conjunto}.parquet"
    ruta = carpeta_particiones / nombre
    return pd.read_parquet(ruta)


def cargar_todas_particiones(carpeta_particiones, carburante, regimen):
    """Carga las tres particiones (train/val/test) de un carburante y régimen.

    Returns:
        dict con keys 'train', 'val', 'test'.
    """
    return {
        conjunto: cargar_particion(carpeta_particiones, carburante, regimen, conjunto)
        for conjunto in ["train", "val", "test"]}


# ─── MODELO BASELINE ────────────────────────────────────────────────────────

def predecir_baseline(df):
    """Predicción del modelo baseline: media móvil de 7 días.

    La columna `precio_mm_7` ya está pre-calculada en el dataset de features
    (generada en el Sprint 3 Fase 2, sin data leakage).

    Args:
        df: DataFrame con la columna 'precio_mm_7'.

    Returns:
        Serie con las predicciones (igual a precio_mm_7).
    """
    return df["precio_mm_7"]

# ─── MODELO PROPHET ─────────────────────────────────────────────────────────

def construir_serie_diaria(df, columna_precio="precio"):
    """Construye la serie diaria nacional a partir de un DataFrame de estaciones.

    Prophet trabaja con series temporales agregadas. Esta función agrega
    las observaciones de todas las estaciones a un único valor diario nacional.

    Args:
        df: DataFrame con columnas 'fecha' y la columna de precio.
        columna_precio: nombre de la columna de precio (por defecto 'precio').

    Returns:
        DataFrame con dos columnas: 'ds' (fecha) y 'y' (precio medio nacional),
        formato esperado por Prophet.
    """
    serie = (df.groupby("fecha")[columna_precio]
               .mean()
               .reset_index()
               .rename(columns={"fecha": "ds", columna_precio: "y"}))
    return serie


def entrenar_prophet(serie_train, **kwargs):
    """Entrena un modelo Prophet sobre una serie temporal.

    Args:
        serie_train: DataFrame con columnas 'ds' (fecha) y 'y' (valor).
        **kwargs: parámetros adicionales para Prophet.

    Returns:
        Modelo Prophet entrenado.
    """
    from prophet import Prophet

    # Configuración por defecto: estacionalidad anual y semanal activas,
    # diaria desactivada (los precios diarios no tienen sub-estructura horaria).
    parametros_defecto = {
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "interval_width": 0.80,  # intervalos de confianza al 80 %
    }
    parametros_defecto.update(kwargs)

    modelo = Prophet(**parametros_defecto)
    modelo.fit(serie_train)
    return modelo


def predecir_prophet(modelo, fechas_objetivo):
    """Genera predicciones con un modelo Prophet entrenado.

    Args:
        modelo: modelo Prophet ya entrenado.
        fechas_objetivo: serie de fechas (Timestamps) para las que predecir.

    Returns:
        DataFrame con la predicción ('yhat'), límite inferior y superior.
    """
    df_futuro = pd.DataFrame({"ds": fechas_objetivo})
    prediccion = modelo.predict(df_futuro)
    return prediccion[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def evaluar_prophet_en_conjunto(modelo, df_conjunto, nombre, columna_precio="precio"):
    """Evalúa un modelo Prophet sobre un conjunto de datos.

    Calcula el precio medio nacional real y la predicción, y devuelve
    las métricas estándar de evaluación.

    Args:
        modelo: modelo Prophet entrenado.
        df_conjunto: DataFrame del conjunto (train/val/test).
        nombre: etiqueta para los resultados.
        columna_precio: columna a evaluar.

    Returns:
        Dict con las métricas estándar.
    """
    serie_real = construir_serie_diaria(df_conjunto, columna_precio)
    prediccion = predecir_prophet(modelo, serie_real["ds"])

    # Unir real y predicho por fecha
    df_eval = serie_real.merge(prediccion, on="ds", how="inner")
    return evaluar(df_eval["y"].values, df_eval["yhat"].values, nombre=nombre)


# ─── PERSISTENCIA DE MODELOS ────────────────────────────────────────────────

def guardar_modelo_prophet(modelo, ruta_salida):
    """Serializa un modelo Prophet a disco.

    Args:
        modelo: modelo Prophet entrenado.
        ruta_salida: Path donde guardar el modelo (.json).
    """
    import json
    from prophet.serialize import model_to_json

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_salida, "w") as f:
        f.write(model_to_json(modelo))


def cargar_modelo_prophet(ruta_entrada):
    """Carga un modelo Prophet serializado.

    Args:
        ruta_entrada: Path al fichero .json del modelo.

    Returns:
        Modelo Prophet cargado.
    """
    from prophet.serialize import model_from_json

    with open(ruta_entrada, "r") as f:
        return model_from_json(f.read())

# ─── MODELO LIGHTGBM ────────────────────────────────────────────────────────

# Features que usaremos como entrada del modelo.
# Excluimos: precio (target), fecha, columnas auxiliares, columnas de otros carburantes.
FEATURES_LIGHTGBM = [
    # Calendario
    "dia_semana", "mes", "dia_del_mes", "semana_del_año","es_fin_semana", "es_festivo_nacional",
    # Régimen
    "regimen", "dias_desde_shock",
    # Lags
    "precio_lag_1", "precio_lag_7", "precio_lag_30",
    # Medias móviles
    "precio_mm_7", "precio_mm_30",
    # Contexto nacional
    "precio_medio_nacional_dia", "diferencial_vs_nacional",
    # Categóricas geográficas
    "Rotulo_normalizado", "Provincia", "IDCCAA",]

# Columnas que LightGBM tratará como categóricas
CATEGORICAS_LIGHTGBM = ["regimen", "Rotulo_normalizado", "Provincia", "IDCCAA","dia_semana", "mes", "es_fin_semana", "es_festivo_nacional",]


def preparar_datos_lightgbm(df, features=FEATURES_LIGHTGBM, target="precio"):
    """Prepara X (features) e y (target) para LightGBM.

    Convierte las columnas categóricas al tipo 'category' de pandas
    (requerido por LightGBM para uso nativo de categóricas).

    Args:
        df: DataFrame con todas las features y el target.
        features: lista de columnas a usar como features.
        target: nombre de la columna objetivo.

    Returns:
        Tupla (X, y) donde X es DataFrame y y es Series.
    """
    X = df[features].copy()
    y = df[target].copy()

    # Convertir categóricas al tipo 'category' de pandas
    for col in CATEGORICAS_LIGHTGBM:
        if col in X.columns:
            X[col] = X[col].astype("category")

    return X, y

def entrenar_lightgbm(X_train, y_train, X_val=None, y_val=None, **kwargs):
    """Entrena un modelo LightGBM con early stopping si hay validation.

    Args:
        X_train: DataFrame de features de entrenamiento.
        y_train: Serie de target de entrenamiento.
        X_val, y_val: opcionales, conjunto de validación para early stopping.
        **kwargs: parámetros adicionales para LightGBM.

    Returns:
        Modelo LightGBM entrenado.
    """
    import lightgbm as lgb

    # Hiperparámetros por defecto (razonables para regresión tabular)
    parametros_defecto = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "n_estimators": 500,
        "verbose": -1,
        "random_state": 42,}
    parametros_defecto.update(kwargs)

    modelo = lgb.LGBMRegressor(**parametros_defecto)

    if X_val is not None and y_val is not None:
        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
        modelo.fit(X_train, y_train,eval_set=[(X_val, y_val)],callbacks=callbacks,)
    else:
        modelo.fit(X_train, y_train)

    return modelo


def predecir_lightgbm(modelo, X):
    """Genera predicciones con un modelo LightGBM entrenado.

    Args:
        modelo: modelo LightGBM entrenado.
        X: DataFrame con las features.

    Returns:
        Array con las predicciones.
    """
    return modelo.predict(X)

def evaluar_lightgbm_en_conjunto(modelo, df_conjunto, nombre, target="precio"):
    """Evalúa un modelo LightGBM en un conjunto de datos.

    Args:
        modelo: modelo LightGBM entrenado.
        df_conjunto: DataFrame del conjunto.
        nombre: etiqueta para los resultados.
        target: columna objetivo.

    Returns:
        Dict con las métricas estándar.
    """
    X, y = preparar_datos_lightgbm(df_conjunto, target=target)
    y_pred = predecir_lightgbm(modelo, X)
    return evaluar(y.values, y_pred, nombre=nombre)


def guardar_modelo_lightgbm(modelo, ruta_salida):
    """Serializa un modelo LightGBM a disco en formato binario nativo.

    Args:
        modelo: modelo LightGBM entrenado.
        ruta_salida: Path donde guardar (.txt).
    """
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    modelo.booster_.save_model(str(ruta_salida))

def cargar_modelo_lightgbm(ruta_entrada):
    """Carga un modelo LightGBM serializado.

    Args:
        ruta_entrada: Path al fichero del modelo.

    Returns:
        Booster de LightGBM cargado.
    """
    import lightgbm as lgb
    return lgb.Booster(model_file=str(ruta_entrada))


def importancia_features_lightgbm(modelo, top_n=20):
    """Devuelve la importancia relativa de las features del modelo.

    Args:
        modelo: modelo LightGBM entrenado.
        top_n: número de features top a devolver.

    Returns:
        DataFrame con feature, importancia y porcentaje.
    """
    importancias = pd.DataFrame({"feature": modelo.feature_name_,"importance": modelo.feature_importances_,})
    importancias = importancias.sort_values("importance", ascending=False).head(top_n)
    importancias["importance_pct"] = (importancias["importance"] / importancias["importance"].sum() * 100).round(2)
    return importancias.reset_index(drop=True)

# ─── MÓDULO DE OPTIMIZACIÓN DE REPOSTAJE ────────────────────────────────────

import math

# Coste de seguridad: margen para preferir repostar hoy sobre arriesgar mañana
MARGEN_SEGURIDAD_CTS = 1.0  # 1 cts/L

def distancia_haversine(lat1, lon1, lat2, lon2):
    """Calcula distancia en km entre dos puntos geográficos (fórmula Haversine).

    Args:
        lat1, lon1: latitud y longitud del primer punto.
        lat2, lon2: latitud y longitud del segundo punto.

    Returns:
        Distancia en kilómetros.
    """
    R = 6371.0  # Radio medio de la Tierra en km
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def encontrar_estaciones_cercanas(df_estaciones, lat_base, lon_base, radio_km,
                                     fecha=None, carburante=None):
    """Filtra estaciones dentro de un radio desde un punto base.

    Args:
        df_estaciones: DataFrame con columnas IDEESS, latitud, longitud,
                        precio_real, precio_predicho, fecha, carburante, marca, provincia.
        lat_base, lon_base: coordenadas del punto base.
        radio_km: radio de búsqueda en kilómetros.
        fecha: filtro opcional por fecha.
        carburante: filtro opcional por carburante.

    Returns:
        DataFrame de estaciones dentro del radio, con columna 'distancia_km'.
    """
    df = df_estaciones.copy()
    
    if fecha is not None:
        df = df[df["fecha"] == fecha]
    if carburante is not None:
        df = df[df["carburante"] == carburante]
    
    # Calcular distancia para cada estación
    df["distancia_km"] = df.apply(lambda r: distancia_haversine(lat_base, lon_base, r["latitud"], r["longitud"]),axis=1,)
    
    # Filtrar por radio
    df = df[df["distancia_km"] <= radio_km].copy()
    df = df.sort_values("precio_real")
    
    return df

def decidir_repostaje(estaciones_hoy, estaciones_manana, litros,
                       consumo_l_km, margen_seguridad_cts=MARGEN_SEGURIDAD_CTS):
    """Decide si repostar HOY o ESPERAR a MAÑANA.

    Args:
        estaciones_hoy: DataFrame de estaciones disponibles hoy.
        estaciones_manana: DataFrame de estaciones disponibles mañana
                            (con precio_predicho).
        litros: cantidad de litros a repostar.
        consumo_l_km: consumo del vehículo en l/km (para calcular coste de desvío).
        margen_seguridad_cts: margen para preferir repostar hoy.

    Returns:
        Dict con la decisión: {accion, dia, estacion_id, marca, precio, coste_total,
                                ahorro_vs_alternativa, distancia_km}
    """
    if len(estaciones_hoy) == 0 and len(estaciones_manana) == 0:
        return None
    
    # Opción HOY: mejor estación de hoy
    if len(estaciones_hoy) > 0:
        mejor_hoy = estaciones_hoy.iloc[0]
        # Coste = litros × precio + coste del desvío (ida y vuelta)
        coste_desvio_hoy = 2 * mejor_hoy["distancia_km"] * consumo_l_km * mejor_hoy["precio_real"]
        coste_hoy = litros * mejor_hoy["precio_real"] + coste_desvio_hoy
    else:
        coste_hoy = float("inf")
        mejor_hoy = None
    
    # Opción MAÑANA: mejor estación predicha de mañana
    if len(estaciones_manana) > 0:
        mejor_manana = estaciones_manana.iloc[0]
        coste_desvio_manana = 2 * mejor_manana["distancia_km"] * consumo_l_km * mejor_manana["precio_predicho"]
        # Añadimos margen de seguridad (el modelo puede equivocarse)
        coste_manana = litros * (mejor_manana["precio_predicho"] + margen_seguridad_cts / 100) + coste_desvio_manana
    else:
        coste_manana = float("inf")
        mejor_manana = None
    
    # Decisión
    if coste_hoy <= coste_manana:
        return {
            "accion": "repostar",
            "dia": "hoy",
            "estacion_id": mejor_hoy["IDEESS"],
            "marca": mejor_hoy["marca"],
            "precio": mejor_hoy["precio_real"],
            "coste_total": coste_hoy,
            "ahorro_vs_alternativa": coste_manana - coste_hoy if coste_manana != float("inf") else 0,
            "distancia_km": mejor_hoy["distancia_km"],}
    else:
        return {
            "accion": "esperar",
            "dia": "mañana",
            "estacion_id": mejor_manana["IDEESS"],
            "marca": mejor_manana["marca"],
            "precio": mejor_manana["precio_predicho"],
            "coste_total": coste_manana,
            "ahorro_vs_alternativa": coste_hoy - coste_manana,
            "distancia_km": mejor_manana["distancia_km"],}


def optimizar_repostaje_estacionario(df_predicciones, lat_base, lon_base,
                                         radio_km, fecha_hoy, fecha_manana,
                                         litros, consumo_l_km, carburante):
    """Optimización completa para un vehículo estacionario.

    Args:
        df_predicciones: DataFrame con todas las predicciones del modelo.
        lat_base, lon_base: ubicación del vehículo.
        radio_km: radio de búsqueda.
        fecha_hoy: fecha actual.
        fecha_manana: fecha del día siguiente.
        litros: litros a repostar.
        consumo_l_km: consumo del vehículo.
        carburante: 'Gasóleo A' o 'Gasolina 95 E5'.

    Returns:
        Dict con la decisión optimizada + información de comparativa.
    """
    estaciones_hoy = encontrar_estaciones_cercanas(df_predicciones, lat_base, lon_base, radio_km,fecha=fecha_hoy, carburante=carburante,)
    estaciones_manana = encontrar_estaciones_cercanas(df_predicciones, lat_base, lon_base, radio_km,fecha=fecha_manana, carburante=carburante,)
    
    decision = decidir_repostaje(estaciones_hoy, estaciones_manana, litros, consumo_l_km)
    
    # Comparativa: ¿cuánto se ahorra vs estación más cara dentro del radio?
    if decision is not None and len(estaciones_hoy) > 0:
        precio_max = estaciones_hoy["precio_real"].max()
        coste_naive = litros * precio_max
        decision["ahorro_vs_naive"] = coste_naive - decision["coste_total"]
        decision["precio_mas_caro_radio"] = precio_max
        decision["n_estaciones_disponibles"] = len(estaciones_hoy)
    
    return decision


def optimizar_repostaje_ruta(df_predicciones, waypoints, fecha,
                                radio_km, litros, consumo_l_km, carburante,
                                nivel_inicial_l=None, capacidad_deposito_l=None):
    """Optimización para vehículo en movimiento con restricción de autonomía.

    Identifica el mejor punto de la ruta para repostar, considerando:
    - Precio de las estaciones en cada waypoint.
    - Coste del desvío.
    - VIABILIDAD: el vehículo solo puede repostar en waypoints alcanzables
      con el combustible disponible.

    Args:
        df_predicciones: DataFrame de predicciones.
        waypoints: lista de tuplas (nombre, lat, lon) representando la ruta.
        fecha: fecha de la operación.
        radio_km: radio de búsqueda por waypoint.
        litros: litros a repostar.
        consumo_l_km: consumo del vehículo en l/km.
        carburante: tipo de carburante.
        nivel_inicial_l: litros disponibles al inicio de la ruta. Si None,
                          se asume capacidad suficiente (sin restricción).
        capacidad_deposito_l: capacidad total del depósito. Si None, sin restricción.

    Returns:
        Dict con la decisión optimizada y análisis de viabilidad.
    """
    # ---- Cálculo de distancias acumuladas entre waypoints ----
    # Factor de corrección Haversine→carretera: las carreteras hacen curvas y la distancia real por carretera es ~18% mayor que la geodésica
    # (Boscoe et al., 2012; valor habitual en estudios logísticos españoles)
    FACTOR_CARRETERA = 1.18
    
    distancias_acumuladas = [0.0]  # waypoint 0 = origen, distancia 0
    for i in range(1, len(waypoints)):
        nombre_prev, lat_prev, lon_prev = waypoints[i - 1]
        nombre_act, lat_act, lon_act = waypoints[i]
        dist_geodesica = distancia_haversine(lat_prev, lon_prev, lat_act, lon_act)
        dist_carretera = dist_geodesica * FACTOR_CARRETERA
        distancias_acumuladas.append(distancias_acumuladas[-1] + dist_carretera)

    # ---- Identificar waypoints viables (alcanzables) ----
    # Si no hay restricción de nivel inicial, todos son viables
    sin_restriccion = nivel_inicial_l is None or consumo_l_km is None
    
    todas_opciones = []
    
    for i, (nombre_wp, lat_wp, lon_wp) in enumerate(waypoints):
        # Calcular si es viable alcanzar este waypoint con el combustible inicial
        distancia_desde_origen = distancias_acumuladas[i]
        combustible_necesario = distancia_desde_origen * consumo_l_km
        
        if not sin_restriccion and combustible_necesario > nivel_inicial_l:
            # No es viable, lo descartamos
            todas_opciones.append({
                "waypoint": nombre_wp,
                "distancia_acumulada_km": distancia_desde_origen,
                "combustible_necesario": combustible_necesario,
                "viable": False,
                "razon_inviable": f"Requiere {combustible_necesario:.0f} L pero solo dispone de {nivel_inicial_l:.0f} L",
            })
            continue
        
        # Buscar estaciones cercanas al waypoint
        estaciones_wp = encontrar_estaciones_cercanas(
            df_predicciones, lat_wp, lon_wp, radio_km,
            fecha=fecha, carburante=carburante,
        )
        if len(estaciones_wp) == 0:
            todas_opciones.append({
                "waypoint": nombre_wp,
                "distancia_acumulada_km": distancia_desde_origen,
                "viable": True,
                "n_estaciones_radio": 0,
                "razon_inviable": "Sin estaciones en radio",
            })
            continue
        
        # Mejor estación del waypoint
        mejor = estaciones_wp.iloc[0]
        coste_desvio = 2 * mejor["distancia_km"] * consumo_l_km * mejor["precio_real"]
        coste_total = litros * mejor["precio_real"] + coste_desvio
        
        todas_opciones.append({
            "waypoint": nombre_wp,
            "distancia_acumulada_km": distancia_desde_origen,
            "combustible_necesario": combustible_necesario,
            "viable": True,
            "estacion_id": mejor["IDEESS"],
            "marca": mejor["marca"],
            "precio": mejor["precio_real"],
            "distancia_km": mejor["distancia_km"],
            "coste_total": coste_total,
            "n_estaciones_radio": len(estaciones_wp),
        })

    # Filtrar opciones viables CON estación encontrada
    opciones_viables = [o for o in todas_opciones 
                         if o.get("viable", False) and "coste_total" in o]
    
    if not opciones_viables:
        return {
            "mejor_opcion": None,
            "todas_opciones": todas_opciones,
            "ahorro_vs_peor_waypoint": 0,
            "error": "Ningún waypoint viable con estaciones",
        }

    # Ordenar por coste total ascendente
    opciones_viables.sort(key=lambda x: x["coste_total"])
    
    return {
        "mejor_opcion": opciones_viables[0],
        "todas_opciones": todas_opciones,  # incluye no viables (para diagnóstico)
        "opciones_viables": opciones_viables,
        "ahorro_vs_peor_waypoint": (
            opciones_viables[-1]["coste_total"] - opciones_viables[0]["coste_total"]
        ),
        "n_viables": len(opciones_viables),
        "n_no_viables": len(todas_opciones) - len(opciones_viables),
    }

# ─── PREDICCIÓN RECURSIVA MULTI-DÍA ─────────────────────────────────────────

def predecir_recursivo_multi_dia(modelo, df_combinado, fechas_futuras,
                                    columna_precio, fecha_corte_shock=None):
    """Predice precios para N días futuros de forma recursiva.
    
    Para cada día futuro:
    1. Genera las features incluyendo predicciones de días anteriores como lags.
    2. Aplica el modelo LightGBM para predecir.
    3. Las predicciones se añaden al dataset para el siguiente paso.
    
    Args:
        modelo: modelo LightGBM (Booster) cargado.
        df_combinado: DataFrame con histórico + días reales recientes.
        fechas_futuras: lista de pd.Timestamp con las fechas a predecir (en orden).
        columna_precio: 'Precio Gasoleo A' o 'Precio Gasolina 95 E5'.
        fecha_corte_shock: opcional, para asignar régimen.
    
    Returns:
        DataFrame con columnas: IDEESS, fecha, precio_predicho, dia_horizonte.
    """
    from src.features import preparar_dataset_carburante
    
    df_trabajo = df_combinado.copy()
    predicciones_acumuladas = []
    
    for i, fecha_objetivo in enumerate(fechas_futuras, 1):
        # 1. Identificar las estaciones que tenían datos el día anterior
        fecha_anterior = fecha_objetivo - pd.Timedelta(days=1)
        estaciones_activas = df_trabajo[
            df_trabajo["fecha"] == fecha_anterior
        ]["IDEESS"].unique()
        
        if len(estaciones_activas) == 0:
            # No hay datos del día anterior, no podemos predecir
            continue
        
        # 2. Crear filas placeholder para el día objetivo (con NaN en precio)
        df_placeholder = df_trabajo[
            df_trabajo["fecha"] == fecha_anterior
        ].copy()
        df_placeholder["fecha"] = fecha_objetivo
        df_placeholder[columna_precio] = np.nan  # se predecirá
        
        # 3. Concatenar al dataset de trabajo
        df_trabajo_extendido = pd.concat([df_trabajo, df_placeholder], ignore_index=True)
        df_trabajo_extendido = df_trabajo_extendido.sort_values(["IDEESS", "fecha"]).reset_index(drop=True)
        
        # 4. Generar features para todo el dataset (incluye el día objetivo)
        df_features = preparar_dataset_carburante(df_trabajo_extendido, columna_precio)
        
        # 5. Filtrar SOLO las filas del día objetivo
        df_dia_objetivo = df_features[df_features["fecha"] == fecha_objetivo].copy()
        
        if len(df_dia_objetivo) == 0:
            continue
        
        # 6. Predecir con LightGBM
        X, _ = preparar_datos_lightgbm(df_dia_objetivo)
        y_pred = modelo.predict(X)
        
        # 7. Añadir las predicciones al df_trabajo original (para el siguiente paso recursivo)
        df_predicciones_paso = df_dia_objetivo[["IDEESS", "fecha"]].copy()
        df_predicciones_paso["precio_predicho"] = y_pred
        df_predicciones_paso["dia_horizonte"] = i
        
        # Información adicional para la pestaña
        df_predicciones_paso = df_predicciones_paso.merge(
            df_dia_objetivo[["IDEESS", "Rotulo_normalizado", "Provincia", 
                              "Latitud", "Longitud (WGS84)"]],
            on="IDEESS",
            how="left",
        )
        df_predicciones_paso.columns = ["IDEESS", "fecha", "precio_predicho", "dia_horizonte",
                                          "marca", "provincia", "latitud", "longitud"]
        
        predicciones_acumuladas.append(df_predicciones_paso)
        
        # Actualizar df_trabajo con la predicción para el siguiente paso recursivo
        # Crear filas con la predicción como "precio real" para que los lags futuros la usen
        df_dia_predicho_para_lags = df_placeholder[df_placeholder["IDEESS"].isin(df_predicciones_paso["IDEESS"])].copy()
        df_dia_predicho_para_lags = df_dia_predicho_para_lags.merge(
            df_predicciones_paso[["IDEESS", "precio_predicho"]],
            on="IDEESS",
            how="left",
        )
        df_dia_predicho_para_lags[columna_precio] = df_dia_predicho_para_lags["precio_predicho"]
        df_dia_predicho_para_lags = df_dia_predicho_para_lags.drop(columns=["precio_predicho"])
        
        df_trabajo = pd.concat([df_trabajo, df_dia_predicho_para_lags], ignore_index=True)
    
    if not predicciones_acumuladas:
        return pd.DataFrame(columns=["IDEESS", "fecha", "precio_predicho", "dia_horizonte",
                                       "marca", "provincia", "latitud", "longitud"])
    
    return pd.concat(predicciones_acumuladas, ignore_index=True)
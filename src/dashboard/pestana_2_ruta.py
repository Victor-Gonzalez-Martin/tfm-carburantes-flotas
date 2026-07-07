"""Pestaña 2 — Planificador de rutas con waypoints."""

import sys
from pathlib import Path
from datetime import date

import pandas as pd
import streamlit as st
import pydeck as pdk

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.modelos import optimizar_repostaje_ruta, distancia_haversine
from src.dashboard.data_loader import (
    cargar_predicciones_validacion,
    obtener_carburantes_disponibles,
)
from src.dashboard.utils import (
    formatear_euros,
    formatear_precio_litro,
    COLOR_PRIMARIO,
    COLOR_ACENTO,
    COLOR_ALERTA,
)


# Rutas predefinidas (corredores RTE-T del MITMA)
RUTAS_PREDEFINIDAS = {
    "Madrid → Barcelona (Corredor Mediterráneo)": [
        ("Madrid",       40.4168, -3.7038),
        ("Guadalajara",  40.6298, -3.1670),
        ("Calatayud",    41.3494, -1.6442),
        ("Zaragoza",     41.6488, -0.8891),
        ("Lleida",       41.6176,  0.6200),
        ("Barcelona",    41.3851,  2.1734),
    ],
    "Madrid → Sevilla (Corredor Atlántico Sur)": [
        ("Madrid",       40.4168, -3.7038),
        ("Toledo",       39.8628, -4.0273),
        ("Ciudad Real",  38.9849, -3.9272),
        ("Córdoba",      37.8882, -4.7794),
        ("Sevilla",      37.3891, -5.9845),
    ],
    "Madrid → Bilbao (Corredor Atlántico Norte)": [
        ("Madrid",            40.4168, -3.7038),
        ("Aranda de Duero",   41.6700, -3.6900),
        ("Burgos",            42.3439, -3.6969),
        ("Vitoria",           42.8467, -2.6716),
        ("Bilbao",            43.2630, -2.9350),
    ],
}


def render():
    st.header("🚛 Planificador de rutas peninsulares")
    st.markdown(
        "Optimiza el repostaje en rutas de larga distancia. El sistema identifica el "
        "**mejor waypoint** para repostar considerando la **autonomía del vehículo** "
        "y descarta automáticamente los puntos inalcanzables."
    )

    df_pred = cargar_predicciones_validacion()
    carburantes = obtener_carburantes_disponibles()

    # Verificar predicciones disponibles
    if "tipo_prediccion" not in df_pred.columns:
        st.warning(
            "⚠️ El dataset de predicciones aún no incluye predicciones futuras. "
            "Pulsa **'Actualizar precios del Ministerio'** en el panel lateral."
        )
        return

    df_reales = df_pred[df_pred["tipo_prediccion"] == "real"]
    df_futuras = df_pred[df_pred["tipo_prediccion"] == "futura"]

    # ─── PANEL DE PARAMETROS ───────────────────────────────────────────
    st.subheader("📍 Configura tu ruta")

    col1, col2 = st.columns([1, 1])

    with col1:
        ruta_nombre = st.selectbox(
            "Ruta",
            options=list(RUTAS_PREDEFINIDAS.keys()),
            index=0,
        )
        waypoints = RUTAS_PREDEFINIDAS[ruta_nombre]
        
        carburante = st.selectbox(
            "Carburante",
            options=carburantes,
            index=0,
            key="carburante_pestana_2",
        )

    with col2:
        litros = st.number_input(
            "Litros a repostar en ruta",
            min_value=50, max_value=600,
            value=400, step=50,
        )
        consumo_l_100km = st.number_input(
            "Consumo del vehículo (L/100km)",
            min_value=20.0, max_value=50.0,
            value=35.0, step=0.5,
            format="%.1f",
            help="35 L/100km típico tráiler articulado",
        )
        consumo_l_km = consumo_l_100km / 100

    # Parámetros de autonomía
    st.markdown("**🔋 Parámetros de autonomía**")
    col_a1, col_a2, col_a3 = st.columns(3)
    
    with col_a1:
        capacidad_deposito = st.number_input(
            "Capacidad depósito (L)",
            min_value=200, max_value=1000,
            value=600, step=50,
        )
    with col_a2:
        nivel_inicial_pct = st.slider(
            "Nivel inicial del depósito (%)",
            min_value=10, max_value=100,
            value=25, step=5,
        )
        nivel_inicial = capacidad_deposito * nivel_inicial_pct / 100
    with col_a3:
        autonomia_inicial = nivel_inicial / consumo_l_km
        st.metric(
            "Autonomía inicial",
            f"{autonomia_inicial:.0f} km",
            help=f"{nivel_inicial:.0f} L disponibles al salir",
        )

    # Selección de día
    st.markdown("**📅 Día de la operación**")
    
    # Construir lista de días disponibles (real + futuros)
    hoy_real = pd.Timestamp(date.today())
    fechas_reales = sorted(df_reales["fecha"].unique()) if not df_reales.empty else []
    fechas_futuras = sorted(df_futuras["fecha"].unique()) if not df_futuras.empty else []
    todas_fechas = list(fechas_reales) + list(fechas_futuras)
    
    def etiqueta_fecha(fecha):
        fecha_ts = pd.Timestamp(fecha)
        diff = (fecha_ts - hoy_real).days
        if diff == -1:
            return f"AYER · {fecha_ts.strftime('%d/%m/%Y')} (dato real)"
        elif diff == 0:
            return f"HOY · {fecha_ts.strftime('%d/%m/%Y')} (dato real)"
        elif diff == 1:
            return f"MAÑANA · {fecha_ts.strftime('%d/%m/%Y')} (predicción)"
        elif diff == 2:
            return f"PASADO MAÑANA · {fecha_ts.strftime('%d/%m/%Y')} (predicción)"
        else:
            return f"{fecha_ts.strftime('%d/%m/%Y')}"
    
    # Por defecto seleccionar el último día con DATO REAL (no predicción futura)
    if fechas_reales:
        ultimo_real = pd.Timestamp(fechas_reales[-1])
        try:
            index_default = list(todas_fechas).index(ultimo_real)
        except (ValueError, IndexError):
            index_default = 0
    else:
        index_default = 0
    
    fecha_ruta = st.selectbox(
        "Fecha",
        options=todas_fechas,
        index=index_default,
        format_func=etiqueta_fecha,
    )
    
    fecha_ruta_ts = pd.Timestamp(fecha_ruta)
    
    # Determinar qué columna de precio usar
    es_futura = fecha_ruta_ts in fechas_futuras
    columna_precio_uso = "precio_predicho" if es_futura else "precio_real"
    
    st.divider()

    # ─── OPTIMIZAR ────────────────────────────────────────────────────
    # Preparar el DataFrame en el formato que espera la función
    df_pred_carb = df_pred[df_pred["carburante"] == carburante].copy()
    df_pred_ruta = df_pred_carb[df_pred_carb["fecha"] == fecha_ruta_ts].copy()
    df_pred_ruta = df_pred_ruta.dropna(subset=[columna_precio_uso])
    
    if df_pred_ruta.empty:
        st.warning(f"⚠️ No hay datos de {carburante} disponibles para {fecha_ruta_ts.date()}.")
        return
    
    # La función espera "precio_real" como columna de precio
    df_pred_ruta_renamed = df_pred_ruta.copy()
    df_pred_ruta_renamed["precio_real"] = df_pred_ruta[columna_precio_uso]
    
    radio_km = 10
    
    decision_ruta = optimizar_repostaje_ruta(
        df_pred_ruta_renamed,
        waypoints,
        fecha_ruta_ts,
        radio_km,
        litros,
        consumo_l_km,
        carburante,
        nivel_inicial_l=nivel_inicial,
        capacidad_deposito_l=capacidad_deposito,
    )
    
    if not decision_ruta or not decision_ruta.get("mejor_opcion"):
        st.error("❌ No se ha podido encontrar una opción viable para esta ruta.")
        return

    # ─── RECOMENDACIÓN ────────────────────────────────────────────────
    mejor = decision_ruta["mejor_opcion"]
    
    st.subheader("✨ Recomendación del sistema")
    st.success(
        f"💡 **Reposta en {mejor['waypoint']}** en la estación **{mejor['marca']}** "
        f"a {formatear_precio_litro(mejor['precio'])}. Coste total estimado: {formatear_euros(mejor['coste_total'])}."
    )

    # KPIs principales
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    
    with col_k1:
        st.metric("📍 Waypoint óptimo", mejor["waypoint"])
    with col_k2:
        st.metric("💰 Coste total", formatear_euros(mejor["coste_total"]))
    with col_k3:
        st.metric(
            "📉 Ahorro vs peor opción",
            formatear_euros(decision_ruta["ahorro_vs_peor_waypoint"]),
            help="Diferencia entre el mejor waypoint y el peor entre los viables",
        )
    with col_k4:
        st.metric(
            "🔋 Viabilidad",
            f"{decision_ruta['n_viables']} / {len(waypoints)} waypoints",
            help=f"{decision_ruta['n_no_viables']} descartados por autonomía",
        )

    st.divider()

    # ─── TABLA DE ANÁLISIS POR WAYPOINT ───────────────────────────────
    st.subheader("📊 Análisis por waypoint")
    st.caption(
        "Distancia acumulada y combustible necesario para alcanzar cada waypoint. "
        "Los waypoints en rojo no son viables con la autonomía disponible."
    )
    
    filas_tabla = []
    for opcion in decision_ruta["todas_opciones"]:
        viable = opcion.get("viable", False) and "coste_total" in opcion
        filas_tabla.append({
            "Waypoint": opcion["waypoint"],
            "Distancia acum. (km)": round(opcion.get("distancia_acumulada_km", 0), 0),
            "Combustible necesario (L)": round(opcion.get("combustible_necesario", 0), 0),
            "Viable": "✓" if viable else "✗",
            "Marca": opcion.get("marca", "—"),
            "Precio (€/L)": round(opcion["precio"], 3) if "precio" in opcion else None,
            "Coste total (€)": round(opcion["coste_total"], 2) if "coste_total" in opcion else None,
        })
    
    df_tabla = pd.DataFrame(filas_tabla)
    
    st.dataframe(
        df_tabla,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Distancia acum. (km)": st.column_config.NumberColumn(format="%.0f km"),
            "Combustible necesario (L)": st.column_config.NumberColumn(format="%.0f L"),
            "Precio (€/L)": st.column_config.NumberColumn(format="%.3f €/L"),
            "Coste total (€)": st.column_config.NumberColumn(format="%.2f €"),
        },
    )

    st.divider()

    # ─── MAPA INTERACTIVO ─────────────────────────────────────────────
    st.subheader("🗺️ Mapa de la ruta")
    st.caption(
        f"Línea azul: ruta completa. Marcadores verdes: waypoints viables. "
        f"Marcador rojo: estación óptima de repostaje."
    )

    # Capa 1: línea de la ruta
    df_ruta_line = pd.DataFrame({
        "path": [[[lon, lat] for nombre, lat, lon in waypoints]],
    })
    
    capa_ruta = pdk.Layer(
        "PathLayer",
        df_ruta_line,
        get_path="path",
        get_color=[26, 79, 139, 200],
        width_min_pixels=4,
        pickable=False,
    )
    
    # Capa 2: marcadores de waypoints
    df_waypoints = pd.DataFrame([
        {
            "nombre": nombre,
            "lat": lat,
            "lon": lon,
            "viable": any(o["waypoint"] == nombre and o.get("viable", False) and "coste_total" in o
                          for o in decision_ruta["todas_opciones"]),
        }
        for nombre, lat, lon in waypoints
    ])
    df_waypoints["color"] = df_waypoints["viable"].apply(
        lambda v: [40, 167, 69, 220] if v else [220, 53, 69, 220]
    )
    
    capa_wp = pdk.Layer(
        "ScatterplotLayer",
        df_waypoints,
        get_position=["lon", "lat"],
        get_radius=5000,
        get_fill_color="color",
        get_line_color=[255, 255, 255, 255],
        line_width_min_pixels=2,
        stroked=True,
        pickable=True,
    )
    
    # Capa 3: estación óptima (destacada)
    estacion_optima = df_pred_ruta[
        df_pred_ruta["IDEESS"] == mejor["estacion_id"]
    ].copy()
    
    if not estacion_optima.empty:
        capa_optima = pdk.Layer(
            "ScatterplotLayer",
            estacion_optima,
            get_position=["longitud", "latitud"],
            get_radius=3000,
            get_fill_color=[255, 140, 0, 240],
            get_line_color=[255, 255, 255, 255],
            line_width_min_pixels=3,
            stroked=True,
            pickable=True,
        )
        capas = [capa_ruta, capa_wp, capa_optima]
    else:
        capas = [capa_ruta, capa_wp]
    
    # Vista centrada en la ruta
    lat_centro = sum(lat for _, lat, _ in waypoints) / len(waypoints)
    lon_centro = sum(lon for _, _, lon in waypoints) / len(waypoints)
    
    # Calcular zoom según extensión
    lat_range = max(lat for _, lat, _ in waypoints) - min(lat for _, lat, _ in waypoints)
    lon_range = max(lon for _, _, lon in waypoints) - min(lon for _, _, lon in waypoints)
    extension = max(lat_range, lon_range)
    
    if extension < 2:
        zoom = 7
    elif extension < 5:
        zoom = 6
    else:
        zoom = 5
    
    view_state = pdk.ViewState(
        latitude=lat_centro,
        longitude=lon_centro,
        zoom=zoom,
        pitch=0,
    )
    
    tooltip = {
        "html": "<b>{nombre}</b>",
        "style": {
            "backgroundColor": "rgba(26, 79, 139, 0.95)",
            "color": "white",
            "padding": "6px 10px",
            "borderRadius": "4px",
        },
    }
    
    deck = pdk.Deck(
        layers=capas,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="light",
    )
    
    st.pydeck_chart(deck)
    
    st.caption(
        f"📊 Sistema RepostaPro · Optimización Madrid → {waypoints[-1][0]} · "
        f"Carburante: {carburante} · Fecha: {etiqueta_fecha(fecha_ruta_ts)}"
    )
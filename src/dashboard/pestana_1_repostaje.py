"""Pestaña 1 — Repostaje óptimo operativo (HOY / MAÑANA / PASADO MAÑANA)."""

import sys
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import streamlit as st
import pydeck as pdk

# Añadir raíz al path
RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.modelos import (
    encontrar_estaciones_cercanas,
)
from src.dashboard.data_loader import (
    cargar_predicciones_validacion,
    obtener_carburantes_disponibles,
)
from src.dashboard.utils import (
    crear_mapa_pydeck_con_radio,
    formatear_euros,
    formatear_precio_litro,
    geocodificar_direccion,
    COLOR_PRIMARIO,
    COLOR_ACENTO,
    COLOR_ALERTA,
)


# Ubicaciones predefinidas
UBICACIONES_PRESET = {
    "Vallecas (Madrid)": (40.3925, -3.6650),
    "Coslada (Polígono logístico)": (40.4239, -3.4836),
    "Centro de Madrid": (40.4168, -3.7038),
    "Madrid - Aeropuerto T4": (40.4936, -3.5676),
    "🔍 Buscar por dirección": "BUSCAR",
    "📍 Personalizado (coordenadas)": None,
}


def render():
    st.header("🎯 Repostaje óptimo desde tu ubicación")
    st.markdown(
        "Compara el precio **real de hoy** con las **predicciones para mañana y pasado mañana**. "
        "El sistema te recomienda el mejor día para repostar."
    )

    # Cargar datos
    df_pred = cargar_predicciones_validacion()
    carburantes = obtener_carburantes_disponibles()
    
    # Verificar que tenemos predicciones futuras
    if "tipo_prediccion" not in df_pred.columns:
        st.warning(
            "⚠️ El dataset de predicciones aún no incluye predicciones futuras. "
            "Pulsa el botón **'Actualizar precios del Ministerio'** en el panel lateral "
            "para descargar datos y generar predicciones para mañana y pasado mañana."
        )
        return
    
    # Detectar días disponibles
    df_reales = df_pred[df_pred["tipo_prediccion"] == "real"].copy()
    df_futuras = df_pred[df_pred["tipo_prediccion"] == "futura"].copy()
    
    if df_reales.empty:
        st.warning("⚠️ No hay datos reales disponibles. Pulsa el botón de actualización.")
        return
    
    # Día HOY = último día con datos reales
    fecha_hoy = df_reales["fecha"].max()
    
    # Días futuros disponibles (mañana, pasado mañana)
    fechas_futuras_disponibles = sorted(df_futuras["fecha"].unique()) if not df_futuras.empty else []

    # ─── PANEL DE PARAMETROS ───────────────────────────────────────────
    st.subheader("📍 Configura tu búsqueda")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        ubicacion_label = st.selectbox(
            "Ubicación base",
            options=list(UBICACIONES_PRESET.keys()),
            index=0,
        )

        valor_preset = UBICACIONES_PRESET[ubicacion_label]

        if valor_preset == "BUSCAR":
            direccion = st.text_input(
                "Dirección",
                value="Calle de Alcalá 50, Madrid",
                help="Ej: 'Paseo de la Castellana 95, Madrid'",
            )
            if direccion:
                coords = geocodificar_direccion(direccion)
                if coords:
                    lat_base, lon_base = coords
                    st.success(f"✓ {lat_base:.4f}, {lon_base:.4f}")
                else:
                    st.error("❌ No se ha encontrado la dirección.")
                    return
            else:
                st.info("Introduce una dirección para buscar.")
                return
        elif valor_preset is None:
            lat_base = st.number_input("Latitud", value=40.4168, format="%.4f")
            lon_base = st.number_input("Longitud", value=-3.7038, format="%.4f")
        else:
            lat_base, lon_base = valor_preset
            st.caption(f"Coordenadas: {lat_base}, {lon_base}")

    with col2:
        carburante = st.selectbox(
            "Carburante",
            options=carburantes,
            index=0,
        )
        radio_km = st.slider(
            "Radio de búsqueda (km)",
            min_value=1, max_value=30,
            value=5, step=1,
        )

    with col3:
        litros = st.number_input(
            "Litros a repostar",
            min_value=10, max_value=1000,
            value=50, step=10,
        )
        consumo_l_100km = st.number_input(
            "Consumo del vehículo (L/100km)",
            min_value=5.0, max_value=50.0,
            value=10.0, step=0.5,
            format="%.1f",
            help="10 L/100km = furgoneta · 28 L/100km = camión rígido · 35 L/100km = tráiler",
        )
        consumo_l_km = consumo_l_100km / 100

    st.divider()

    # ─── ANALIZAR HOY / MAÑANA / PASADO MAÑANA ─────────────────────────
    
    # Filtrar carburante
    df_pred_carb = df_pred[df_pred["carburante"] == carburante].copy()
    
    # Función auxiliar para obtener el mejor precio + estación de un día
    def analizar_dia(fecha_objetivo, columna_precio):
        """Encuentra la mejor estación en el radio para una fecha y columna de precio."""
        df_dia = df_pred_carb[df_pred_carb["fecha"] == fecha_objetivo].copy()
        if df_dia.empty:
            return None
        df_dia = df_dia.rename(columns={columna_precio: "precio_uso"})
        df_dia = df_dia.dropna(subset=["precio_uso"])
        if df_dia.empty:
            return None
        # Calcular distancia
        from src.modelos import distancia_haversine
        df_dia["distancia_km"] = df_dia.apply(
            lambda r: distancia_haversine(lat_base, lon_base, r["latitud"], r["longitud"]),
            axis=1,
        )
        df_dia = df_dia[df_dia["distancia_km"] <= radio_km].copy()
        if df_dia.empty:
            return None
        df_dia = df_dia.sort_values("precio_uso")
        mejor = df_dia.iloc[0]
        # Coste = litros × precio + coste desvío (ida y vuelta)
        coste_desvio = 2 * mejor["distancia_km"] * consumo_l_km * mejor["precio_uso"]
        coste_total = litros * mejor["precio_uso"] + coste_desvio
        return {
            "df_estaciones": df_dia,
            "mejor": mejor,
            "coste_total": coste_total,
            "precio": mejor["precio_uso"],
            "marca": mejor["marca"],
            "distancia_km": mejor["distancia_km"],
            "n_estaciones": len(df_dia),
        }
    
    # Analizar cada día disponible
    resultado_hoy = analizar_dia(fecha_hoy, "precio_real")
    
    resultados_futuros = []
    for i, fecha_fut in enumerate(fechas_futuras_disponibles, 1):
        res = analizar_dia(fecha_fut, "precio_predicho")
        if res:
            res["fecha"] = fecha_fut
            res["dia_horizonte"] = i
            resultados_futuros.append(res)
    
    if not resultado_hoy:
        st.warning(f"⚠️ No se encontraron estaciones de {carburante} en un radio de {radio_km} km.")
        return
    
    # ─── DECISIÓN ÓPTIMA ──────────────────────────────────────────────
    todos_resultados = [{"fecha": fecha_hoy, "dia_horizonte": 0, **resultado_hoy}] + resultados_futuros
    mejor_dia = min(todos_resultados, key=lambda r: r["coste_total"])
    
    # Etiquetas
    def etiqueta_dia(dia_h, fecha):
        """Etiqueta operativa según si la fecha es ayer/hoy/mañana del usuario.
        
        El Ministerio publica el día anterior, por lo que:
        - dia_h=0 (último dato real) = AYER del usuario
        - dia_h=1 (predicción) = HOY del usuario  
        - dia_h=2 (predicción) = MAÑANA del usuario
        """
        hoy_real = pd.Timestamp(date.today())
        fecha_ts = pd.Timestamp(fecha)
        
        diferencia_dias = (fecha_ts - hoy_real).days
        
        if diferencia_dias == -1:
            return f"AYER ({fecha_ts.strftime('%d/%m')}) · dato real"
        elif diferencia_dias == 0:
            return f"HOY ({fecha_ts.strftime('%d/%m')}) · dato real"
        elif diferencia_dias == 1:
            return f"MAÑANA ({fecha_ts.strftime('%d/%m')}) · predicción"
        elif diferencia_dias == 2:
            return f"PASADO MAÑANA ({fecha_ts.strftime('%d/%m')}) · predicción"
        elif diferencia_dias < 0:
            return f"Hace {abs(diferencia_dias)} días ({fecha_ts.strftime('%d/%m')})"
        else:
            return f"+{diferencia_dias} días ({fecha_ts.strftime('%d/%m')})"
    
    mejor_etiqueta = etiqueta_dia(mejor_dia["dia_horizonte"], mejor_dia["fecha"])
    
    # ─── RECOMENDACIÓN DESTACADA ──────────────────────────────────────
    st.subheader("✨ Recomendación del sistema")
    # Determinar texto de recomendación según el día (operativo)
    hoy_real = pd.Timestamp(date.today())
    diferencia_dias_mejor = (pd.Timestamp(mejor_dia["fecha"]) - hoy_real).days
    
    if diferencia_dias_mejor == -1:
        recomendacion_texto = (
            f"⚠️ **EL MEJOR PRECIO ESTUVO AYER** en **{mejor_dia['marca']}** "
            f"a {formatear_precio_litro(mejor_dia['precio'])}. "
            f"Las predicciones para hoy y mañana son menos favorables."
        )
        recomendacion_color = "warning"
    elif diferencia_dias_mejor == 0:
        recomendacion_texto = (
            f"💡 **REPOSTA HOY** en **{mejor_dia['marca']}** "
            f"al precio predicho de {formatear_precio_litro(mejor_dia['precio'])}"
        )
        recomendacion_color = "success"
    else:
        recomendacion_texto = (
            f"⏰ **ESPERA A {mejor_etiqueta}** y reposta en **{mejor_dia['marca']}** "
            f"al precio predicho de {formatear_precio_litro(mejor_dia['precio'])}"
        )
        recomendacion_color = "info"
    
    if recomendacion_color == "success":
        st.success(recomendacion_texto)
    elif recomendacion_color == "warning":
        st.warning(recomendacion_texto)
    else:
        st.info(recomendacion_texto)
    
    # ─── KPIs COMPARATIVOS POR DÍA ────────────────────────────────────
    st.subheader("📊 Comparativa por día")
    st.caption("Coste total de repostar los litros indicados en la mejor estación de cada día.")
    
    n_dias = len(todos_resultados)
    cols_dia = st.columns(n_dias)
    
    for i, (col, res) in enumerate(zip(cols_dia, todos_resultados)):
        es_mejor = res["dia_horizonte"] == mejor_dia["dia_horizonte"]
        etiqueta = etiqueta_dia(res["dia_horizonte"], res["fecha"])
        
        # Calcular diferencia vs HOY
        if res["dia_horizonte"] == 0:
            delta = None
        else:
            delta = res["coste_total"] - resultado_hoy["coste_total"]
            delta_str = formatear_euros(delta)
            if delta < 0:
                delta_str = f"-{formatear_euros(abs(delta))}"
        
        with col:
            etiqueta_display = f"{'⭐ ' if es_mejor else ''}{etiqueta}"
            if res["dia_horizonte"] == 0:
                st.metric(
                    etiqueta_display,
                    formatear_euros(res["coste_total"]),
                    help=f"Precio: {formatear_precio_litro(res['precio'])} en {res['marca']}",
                )
            else:
                st.metric(
                    etiqueta_display,
                    formatear_euros(res["coste_total"]),
                    delta=delta_str if delta else None,
                    delta_color="inverse" if delta else "off",
                    help=f"Predicción: {formatear_precio_litro(res['precio'])} en {res['marca']}",
                )
            
            # Info adicional
            st.markdown(
                f"<small>📍 {res['marca']}<br/>"
                f"💰 {formatear_precio_litro(res['precio'])}<br/>"
                f"📏 {res['distancia_km']:.2f} km</small>",
                unsafe_allow_html=True,
            )
    
    # Ahorro total
    coste_peor = max(r["coste_total"] for r in todos_resultados)
    coste_mejor = mejor_dia["coste_total"]
    ahorro_temporal = coste_peor - coste_mejor
    
    if ahorro_temporal > 0.01:
        st.success(
            f"💰 **Ahorro por elegir el mejor día**: {formatear_euros(ahorro_temporal)} "
            f"({(ahorro_temporal / coste_peor * 100):.1f}% sobre repostar el peor día)"
        )
    
    st.divider()

    # ─── MAPA INTERACTIVO ─────────────────────────────────────────────
    st.subheader(f"🗺️ Mapa de estaciones — Precios de {mejor_etiqueta}")
    st.caption(
        f"Marcadores coloreados por precio: 🟢 verde = más barato, 🔴 rojo = más caro. "
        f"Pasa el ratón sobre cualquier marcador para ver detalles."
    )

    df_mapa = mejor_dia["df_estaciones"].copy()
    df_mapa["precio_real"] = df_mapa["precio_uso"].round(3)

    mapa = crear_mapa_pydeck_con_radio(
        df_mapa,
        lat_base, lon_base, radio_km,
        zoom=12 if radio_km <= 5 else 11 if radio_km <= 15 else 10,
    )
    st.pydeck_chart(mapa)

    st.divider()

    # ─── TABLA TOP 10 ESTACIONES ─────────────────────────────────────
    st.subheader(f"📋 Top 10 estaciones más baratas — {mejor_etiqueta}")

    df_top = mejor_dia["df_estaciones"].head(10)[
        ["marca", "provincia", "precio_uso", "distancia_km", "IDEESS"]
    ].copy()
    df_top.columns = ["Marca", "Provincia", "Precio (€/L)", "Distancia (km)", "ID Estación"]
    df_top["Precio (€/L)"] = df_top["Precio (€/L)"].round(3)
    df_top["Distancia (km)"] = df_top["Distancia (km)"].round(2)
    df_top.index = range(1, len(df_top) + 1)

    st.dataframe(
        df_top,
        use_container_width=True,
        column_config={
            "Precio (€/L)": st.column_config.NumberColumn(format="%.3f €/L"),
            "Distancia (km)": st.column_config.NumberColumn(format="%.2f km"),
        },
    )

    st.caption(
        f"📊 Análisis sobre {mejor_dia['n_estaciones']} estaciones de {carburante} "
        f"en un radio de {radio_km} km. Sistema RepostaPro · Modelo LightGBM pre-shock."
    )
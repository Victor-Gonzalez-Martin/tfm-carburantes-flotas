"""Pestaña 3 — Análisis económico para tu flota."""

import sys
from pathlib import Path
from datetime import date

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.modelos import encontrar_estaciones_cercanas
from src.dashboard.data_loader import (
    cargar_predicciones_validacion,
    obtener_carburantes_disponibles,
)
from src.dashboard.utils import (
    formatear_euros,
    COLOR_PRIMARIO,
    COLOR_ACENTO,
    COLOR_ALERTA,
)


# Perfiles típicos del sector logístico español (datos MITMA Observatorio)
PERFILES_TIPO = {
    "Furgoneta de reparto urbano": {
        "consumo_l_100km": 10.0,
        "litros_repostaje": 50,
        "frecuencia_dias": 3,
    },
    "Camión rígido (regional)": {
        "consumo_l_100km": 28.0,
        "litros_repostaje": 150,
        "frecuencia_dias": 1,
    },
    "Tráiler articulado (larga distancia)": {
        "consumo_l_100km": 35.0,
        "litros_repostaje": 400,
        "frecuencia_dias": 2,
    },
    "Personalizado": None,
}

# Ubicaciones base predefinidas
UBICACIONES_BASE = {
    "Vallecas (Madrid)": (40.3925, -3.6650),
    "Coslada (Polígono logístico)": (40.4239, -3.4836),
    "Mercamadrid": (40.3789, -3.6700),
    "Madrid centro": (40.4168, -3.7038),
    "Personalizado": None,
}

# Flotas de referencia del Sprint 4
FLOTAS_REFERENCIA = pd.DataFrame([
    {
        "Flota": "Mensajería urbana\n(30 furgonetas)",
        "Vehículos": 30,
        "Ahorro mensual (€)": 2509,
        "Ahorro anual (€)": 30107,
        "% sobre combustible": 14.7,
    },
    {
        "Flota": "Transporte interurbano\n(15 camiones)",
        "Vehículos": 15,
        "Ahorro mensual (€)": 16164,
        "Ahorro anual (€)": 193973,
        "% sobre combustible": 21.1,
    },
    {
        "Flota": "Largo recorrido\n(8 tráileres)",
        "Vehículos": 8,
        "Ahorro mensual (€)": 6356,
        "Ahorro anual (€)": 76276,
        "% sobre combustible": 8.5,
    },
])


def render():
    st.header("💰 Análisis económico para tu flota")
    st.markdown(
        "Configura los parámetros operacionales de **tu flota** y descubre cuánto te "
        "ahorraría implementar el sistema RepostaPro. Los resultados se comparan con "
        "las tres flotas tipo del estudio del Sprint 4."
    )

    df_pred = cargar_predicciones_validacion()
    carburantes = obtener_carburantes_disponibles()

    if "tipo_prediccion" not in df_pred.columns:
        st.warning(
            "⚠️ Pulsa **'Actualizar precios del Ministerio'** en el sidebar primero."
        )
        return

    # ─── CONFIGURACIÓN DE LA FLOTA ────────────────────────────────────
    st.subheader("🚚 Configura tu flota")

    col1, col2 = st.columns([1, 1])

    with col1:
        perfil_seleccionado = st.selectbox(
            "Perfil de vehículo",
            options=list(PERFILES_TIPO.keys()),
            index=0,
            key="perfil_flota_p3",
        )
        
        perfil_valores = PERFILES_TIPO[perfil_seleccionado]
        if perfil_valores is None:
            perfil_valores = {"consumo_l_100km": 15.0, "litros_repostaje": 100, "frecuencia_dias": 2}
        
        n_vehiculos = st.number_input(
            "Número de vehículos en la flota",
            min_value=1, max_value=500,
            value=20, step=1,
            key="n_vehiculos_p3",
        )
        
        carburante = st.selectbox(
            "Carburante",
            options=carburantes,
            index=0,
            key="carburante_p3",
        )

    with col2:
        consumo_l_100km = st.number_input(
            "Consumo (L/100km)",
            min_value=5.0, max_value=50.0,
            value=float(perfil_valores["consumo_l_100km"]), step=0.5,
            format="%.1f",
            key="consumo_p3",
        )
        consumo_l_km = consumo_l_100km / 100
        
        litros_repostaje = st.number_input(
            "Litros por repostaje",
            min_value=20, max_value=600,
            value=int(perfil_valores["litros_repostaje"]), step=10,
            key="litros_p3",
        )
        
        frecuencia_repostaje = st.number_input(
            "Frecuencia de repostaje (días)",
            min_value=1, max_value=14,
            value=int(perfil_valores["frecuencia_dias"]), step=1,
            help="Cada cuántos días cada vehículo necesita repostar",
            key="frecuencia_p3",
        )

    # Ubicación base
    st.markdown("**📍 Ubicación base de la flota**")
    col_u1, col_u2 = st.columns([1, 1])
    
    with col_u1:
        ubicacion_label = st.selectbox(
            "Ubicación base",
            options=list(UBICACIONES_BASE.keys()),
            index=0,
            key="ubicacion_p3",
        )
        valor_preset = UBICACIONES_BASE[ubicacion_label]
        
        if valor_preset is None:
            lat_base = st.number_input("Latitud", value=40.4168, format="%.4f", key="lat_p3")
            lon_base = st.number_input("Longitud", value=-3.7038, format="%.4f", key="lon_p3")
        else:
            lat_base, lon_base = valor_preset
            st.caption(f"Coordenadas: {lat_base}, {lon_base}")
    
    with col_u2:
        radio_km = st.slider(
            "Radio de búsqueda de estaciones (km)",
            min_value=1, max_value=30,
            value=15, step=1,
            key="radio_p3",
        )
        dias_operacion_mes = st.number_input(
            "Días operacionales al mes",
            min_value=1, max_value=31,
            value=22, step=1,
            key="dias_mes_p3",
        )

    st.divider()

    # ─── CÁLCULO DEL AHORRO ──────────────────────────────────────────
    
    # Usamos el último día real para el análisis
    df_reales = df_pred[df_pred["tipo_prediccion"] == "real"]
    fecha_analisis = df_reales["fecha"].max()
    
    # Estaciones en el radio
    df_pred_carb = df_pred[
        (df_pred["carburante"] == carburante) &
        (df_pred["fecha"] == fecha_analisis)
    ].copy()
    df_pred_carb = df_pred_carb.dropna(subset=["precio_real"])
    
    df_estaciones = encontrar_estaciones_cercanas(
        df_pred_carb, lat_base, lon_base, radio_km,
    )
    
    if df_estaciones.empty:
        st.warning(
            f"⚠️ No se encontraron estaciones de {carburante} en un radio de "
            f"{radio_km} km. Prueba a ampliar el radio."
        )
        return
    
    # Cálculos económicos
    precio_minimo = df_estaciones["precio_real"].min()
    precio_maximo = df_estaciones["precio_real"].max()
    spread = precio_maximo - precio_minimo
    
    # Operaciones mensuales
    operaciones_mes_vehiculo = dias_operacion_mes / frecuencia_repostaje
    operaciones_mes_flota = operaciones_mes_vehiculo * n_vehiculos
    
    # Litros mensuales
    litros_mes_flota = operaciones_mes_flota * litros_repostaje
    
    # Ahorro estimado (precio_max - precio_min) * litros
    ahorro_por_operacion = spread * litros_repostaje
    ahorro_mensual = ahorro_por_operacion * operaciones_mes_flota
    ahorro_anual = ahorro_mensual * 12
    
    # Gasto en combustible
    precio_medio = df_estaciones["precio_real"].mean()
    gasto_mensual_combustible = litros_mes_flota * precio_medio
    gasto_anual_combustible = gasto_mensual_combustible * 12
    
    # % sobre combustible
    pct_ahorro = (ahorro_mensual / gasto_mensual_combustible * 100) if gasto_mensual_combustible > 0 else 0
    
    # Ahorro por vehículo
    ahorro_anual_por_vehiculo = ahorro_anual / n_vehiculos if n_vehiculos > 0 else 0

    # ─── KPIs PRINCIPALES ─────────────────────────────────────────────
    st.subheader("📊 Ahorro estimado para tu flota")
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    
    with col_kpi1:
        st.metric(
            "💵 Ahorro mensual",
            formatear_euros(ahorro_mensual),
            help=f"Sobre {operaciones_mes_flota:.0f} operaciones/mes",
        )
    with col_kpi2:
        st.metric(
            "💰 Ahorro anual",
            formatear_euros(ahorro_anual),
        )
    with col_kpi3:
        st.metric(
            "📈 % sobre combustible",
            f"{pct_ahorro:.1f}%",
            help="Ahorro mensual sobre gasto en combustible",
        )
    with col_kpi4:
        st.metric(
            "🚛 Ahorro por vehículo/año",
            formatear_euros(ahorro_anual_por_vehiculo),
        )

    # ─── DESGLOSE ─────────────────────────────────────────────────────
    st.markdown("**🔍 Desglose del cálculo**")
    
    col_d1, col_d2, col_d3 = st.columns(3)
    
    with col_d1:
        st.markdown(
            f"**Operaciones**  \n"
            f"• Por vehículo/mes: **{operaciones_mes_vehiculo:.1f}**  \n"
            f"• Flota total/mes: **{operaciones_mes_flota:.0f}**  \n"
            f"• Litros mes/flota: **{litros_mes_flota:,.0f} L**"
        )
    
    with col_d2:
        st.markdown(
            f"**Precios en radio ({len(df_estaciones)} estaciones)**  \n"
            f"• Mínimo: **{precio_minimo:.3f} €/L**  \n"
            f"• Máximo: **{precio_maximo:.3f} €/L**  \n"
            f"• Spread: **{spread*100:.1f} cts/L**"
        )
    
    with col_d3:
        st.markdown(
            f"**Gasto**  \n"
            f"• Mensual combustible: **{formatear_euros(gasto_mensual_combustible)}**  \n"
            f"• Anual combustible: **{formatear_euros(gasto_anual_combustible)}**  \n"
            f"• Spread por operación: **{formatear_euros(ahorro_por_operacion)}**"
        )

    st.divider()

    # ─── COMPARATIVA CON FLOTAS DEL ESTUDIO ──────────────────────────
    st.subheader("📈 Comparativa con las flotas del estudio")
    st.caption(
        "Comparación de tu flota personalizada con las 3 flotas-tipo analizadas "
        "en el Sprint 4 del TFM (datos MITMA Observatorio de Costes)."
    )
    
    # Crear DataFrame con tu flota + las 3 de referencia
    tu_flota = pd.DataFrame([{
        "Flota": f"TU FLOTA\n({n_vehiculos} vehículos)",
        "Vehículos": n_vehiculos,
        "Ahorro mensual (€)": ahorro_mensual,
        "Ahorro anual (€)": ahorro_anual,
        "% sobre combustible": pct_ahorro,
    }])
    
    df_comparativa = pd.concat([tu_flota, FLOTAS_REFERENCIA], ignore_index=True)
    
    # Gráfico de barras
    colores_barras = [COLOR_ACENTO] + [COLOR_PRIMARIO] * 3
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_comparativa["Flota"],
        y=df_comparativa["Ahorro anual (€)"],
        marker_color=colores_barras,
        text=df_comparativa["Ahorro anual (€)"].apply(lambda x: formatear_euros(x)),
        textposition="auto",
    ))
    fig.update_layout(
        title="Ahorro anual estimado por flota",
        yaxis_title="Ahorro anual (€)",
        xaxis_title="",
        showlegend=False,
        height=450,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla comparativa
    st.markdown("**Tabla comparativa**")
    df_tabla = df_comparativa.copy()
    df_tabla["Ahorro mensual (€)"] = df_tabla["Ahorro mensual (€)"].round(0).astype(int)
    df_tabla["Ahorro anual (€)"] = df_tabla["Ahorro anual (€)"].round(0).astype(int)
    df_tabla["% sobre combustible"] = df_tabla["% sobre combustible"].round(1)
    
    st.dataframe(
        df_tabla,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ahorro mensual (€)": st.column_config.NumberColumn(format="%d €"),
            "Ahorro anual (€)": st.column_config.NumberColumn(format="%d €"),
            "% sobre combustible": st.column_config.NumberColumn(format="%.1f %%"),
        },
    )

    st.caption(
        f"📊 Sistema RepostaPro · Datos del Ministerio actualizados al "
        f"{fecha_analisis.strftime('%d/%m/%Y')} · "
        f"Análisis sobre {len(df_estaciones)} estaciones de {carburante} en radio de {radio_km} km."
    )
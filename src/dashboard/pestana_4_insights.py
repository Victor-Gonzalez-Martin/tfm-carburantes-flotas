"""Pestaña 4 — Insights, métricas del modelo y análisis histórico interactivo."""

import sys
from pathlib import Path
from datetime import date, timedelta
import logging

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.dashboard.data_loader import (
    cargar_predicciones_validacion,
    CARPETA_RESULTADOS,
    CARPETA_VALIDACION,
    RAIZ_PROYECTO,
)
from src.dashboard.utils import (
    COLOR_PRIMARIO,
    COLOR_ACENTO,
    COLOR_ALERTA,
    formatear_euros,
)


def render():
    st.header("📊 Insights y métricas del modelo")
    st.markdown(
        "Resultados técnicos del modelo de producción, evolución del análisis "
        "metodológico y consulta interactiva del histórico del Ministerio."
    )
    
    # Pestañas internas
    tab_a, tab_b, tab_c, tab_d = st.tabs([
        "🎯 Modelo de producción",
        "🔬 Validación walking-forward",
        "📉 Comparativa de modelos",
        "📅 Histórico interactivo",
    ])
    
    with tab_a:
        render_modelo_produccion()
    
    with tab_b:
        render_walking_forward()
    
    with tab_c:
        render_comparativa()
    
    with tab_d:
        render_historico()


def render_modelo_produccion():
    """Sección con métricas del modelo final + decisión metodológica."""
    st.subheader("Modelo de producción: LightGBM pre-shock")
    
    st.markdown("""
    El modelo final del sistema RepostaPro es un **LightGBM entrenado con datos 
    del régimen pre-shock geopolítico** (enero 2024 - febrero 2026). 
    
    Esta decisión metodológica se basa en el experimento de **validación 
    walking-forward** del Sprint 4, que reveló que el mercado de carburantes 
    post-shock ha evolucionado hacia un régimen estabilizado con precios 
    cercanos al pre-shock.
    """)
    
    # KPIs del modelo
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "MAE Gasóleo A",
            "2,09 cts/L",
            help="Error absoluto medio sobre validación walking-forward 15-21 jun 2026",
        )
    with col2:
        st.metric(
            "MAE Gasolina 95",
            "0,67 cts/L",
            help="Error absoluto medio sobre validación walking-forward",
        )
    with col3:
        st.metric(
            "Período entrenamiento",
            "Ene 2024 - Feb 2026",
            help="Régimen pre-shock (antes del structural break del 1 marzo 2026)",
        )
    with col4:
        st.metric(
            "Estaciones cubiertas",
            "~11.500",
            help="Estaciones de servicio activas en España",
        )
    
    st.divider()
    
    # Decisión metodológica
    st.markdown("### 🔑 Decisión metodológica clave")
    st.info("""
    **¿Por qué el modelo pre-shock supera al post-shock?**
    
    El experimento de validación walking-forward sobre datos reales del periodo 
    15-21 junio 2026 reveló que:
    
    - El modelo **pre-shock** obtuvo un MAE de **2,09 cts/L** en Gasóleo A.
    - El modelo **post-shock** (entrenado abr 2026) obtuvo un MAE de **13,28 cts/L**.
    
    El régimen post-shock tras el shock geopolítico (marzo 2026, Gasóleo A ~1,80 €/L) 
    NO refleja el régimen actual (jun 2026, Gasóleo A ~1,55 €/L). El mercado ha 
    convergido hacia un régimen 'post-shock estabilizado' cuyos precios se sitúan 
    en el rango cubierto por el modelo pre-shock, lo que explica su superior precisión.
    """)
    
    # Features clave
    st.markdown("### 🧩 Features más relevantes del modelo")
    
    features_data = pd.DataFrame([
        {"Feature": "precio_lag_1", "Importancia": 78, "Descripción": "Precio del día anterior"},
        {"Feature": "precio_lag_7", "Importancia": 8, "Descripción": "Precio de hace 7 días"},
        {"Feature": "precio_mm_7", "Importancia": 5, "Descripción": "Media móvil 7 días"},
        {"Feature": "diferencial_vs_nacional", "Importancia": 3, "Descripción": "Diferencia con la media nacional"},
        {"Feature": "Rotulo (marca)", "Importancia": 2, "Descripción": "Marca de la estación"},
        {"Feature": "Provincia", "Importancia": 2, "Descripción": "Provincia de la estación"},
        {"Feature": "Otras (12 features)", "Importancia": 2, "Descripción": "Resto de variables"},
    ])
    
    fig_features = go.Figure()
    fig_features.add_trace(go.Bar(
        x=features_data["Importancia"],
        y=features_data["Feature"],
        orientation="h",
        marker_color=COLOR_PRIMARIO,
        text=features_data["Importancia"].apply(lambda x: f"{x}%"),
        textposition="auto",
    ))
    fig_features.update_layout(
        title="Importancia relativa de las features (Gasóleo A pre-shock)",
        xaxis_title="Importancia (%)",
        yaxis=dict(autorange="reversed"),
        height=350,
        margin=dict(l=180, r=40, t=60, b=40),
    )
    st.plotly_chart(fig_features, use_container_width=True)


def render_walking_forward():
    """Sección con la validación walking-forward."""
    st.subheader("Validación walking-forward (15-21 jun 2026)")
    
    st.markdown("""
    Validación operacional con datos reales descargados del Ministerio para 
    el periodo posterior al cierre del entrenamiento. Comparativa de los 
    modelos pre-shock vs post-shock por carburante.
    """)
    
    # Datos del experimento walking-forward
    datos_wf = {
        "Fecha": pd.date_range("2026-06-15", "2026-06-21"),
        "Gasóleo A · pre-shock": [4.44, 3.30, 1.97, 1.27, 1.18, 1.05, 1.23],
        "Gasóleo A · post-shock": [10.98, 11.79, 12.72, 13.84, 14.55, 14.74, 14.50],
        "Gasolina 95 · pre-shock": [0.37, 0.55, 0.74, 0.86, 0.91, 0.71, 0.51],
        "Gasolina 95 · post-shock": [0.78, 1.36, 2.06, 2.91, 3.36, 4.13, 4.20],
    }
    df_wf = pd.DataFrame(datos_wf)
    
    # 2 gráficas (Gasóleo y Gasolina)
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(
            x=df_wf["Fecha"], y=df_wf["Gasóleo A · pre-shock"],
            mode="lines+markers", name="Pre-shock (PRODUCCIÓN)",
            line=dict(color=COLOR_PRIMARIO, width=3),
            marker=dict(size=10),
        ))
        fig_g.add_trace(go.Scatter(
            x=df_wf["Fecha"], y=df_wf["Gasóleo A · post-shock"],
            mode="lines+markers", name="Post-shock (descartado)",
            line=dict(color="lightblue", width=2, dash="dash"),
            marker=dict(size=8),
        ))
        fig_g.update_layout(
            title="Gasóleo A — MAE por día (cts/L)",
            xaxis_title="Fecha",
            yaxis_title="MAE (céntimos por litro)",
            height=400,
            hovermode="x unified",
        )
        st.plotly_chart(fig_g, use_container_width=True)
    
    with col_g2:
        fig_gl = go.Figure()
        fig_gl.add_trace(go.Scatter(
            x=df_wf["Fecha"], y=df_wf["Gasolina 95 · pre-shock"],
            mode="lines+markers", name="Pre-shock (PRODUCCIÓN)",
            line=dict(color=COLOR_ALERTA, width=3),
            marker=dict(size=10),
        ))
        fig_gl.add_trace(go.Scatter(
            x=df_wf["Fecha"], y=df_wf["Gasolina 95 · post-shock"],
            mode="lines+markers", name="Post-shock (descartado)",
            line=dict(color="lightsalmon", width=2, dash="dash"),
            marker=dict(size=8),
        ))
        fig_gl.update_layout(
            title="Gasolina 95 E5 — MAE por día (cts/L)",
            xaxis_title="Fecha",
            yaxis_title="MAE (céntimos por litro)",
            height=400,
            hovermode="x unified",
        )
        st.plotly_chart(fig_gl, use_container_width=True)
    
    st.info("""
    **Hallazgo principal**: El MAE del modelo pre-shock disminuye consistentemente 
    día a día (Gasóleo A: 4,44 cts el 15 jun → 1,23 cts el 21 jun). Este patrón 
    se explica por la convergencia del precio real hacia el rango de entrenamiento 
    del modelo, lo que valida la decisión de adoptarlo como modelo de producción.
    """)


def render_comparativa():
    """Comparativa de las 3 familias de modelos."""
    st.subheader("Comparativa de modelos en TEST (Sprint 3 Fase 5)")
    
    st.markdown("""
    Comparativa del Sprint 3 con los datos de test originales (validación interna 
    al cierre del histórico de entrenamiento). LightGBM es el modelo seleccionado 
    como familia para producción.
    """)
    
    datos_comp = pd.DataFrame([
        {"Modelo": "Baseline (estación)", "Gasóleo A · pre": 0.4, "Gasolina 95 · pre": 0.38,
         "Gasóleo A · post": 1.79, "Gasolina 95 · post": 1.29},
        {"Modelo": "Prophet", "Gasóleo A · pre": 8.75, "Gasolina 95 · pre": 1.43,
         "Gasóleo A · post": 17.31, "Gasolina 95 · post": 13.46},
        {"Modelo": "LightGBM", "Gasóleo A · pre": 0.49, "Gasolina 95 · pre": 0.91,
         "Gasóleo A · post": 6.64, "Gasolina 95 · post": 0.46},
    ])
    
    # Gráfico de barras agrupadas
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name="Gasóleo A · pre-shock", x=datos_comp["Modelo"], y=datos_comp["Gasóleo A · pre"],
        marker_color="#1A4F8B",
    ))
    fig.add_trace(go.Bar(
        name="Gasóleo A · post-shock", x=datos_comp["Modelo"], y=datos_comp["Gasóleo A · post"],
        marker_color="#7BAFD4",
    ))
    fig.add_trace(go.Bar(
        name="Gasolina 95 · pre-shock", x=datos_comp["Modelo"], y=datos_comp["Gasolina 95 · pre"],
        marker_color="#DC3545",
    ))
    fig.add_trace(go.Bar(
        name="Gasolina 95 · post-shock", x=datos_comp["Modelo"], y=datos_comp["Gasolina 95 · post"],
        marker_color="#F5A6A6",
    ))
    
    fig.update_layout(
        title="MAE (céntimos por litro) por modelo, carburante y régimen",
        yaxis_title="MAE (cts/L)",
        barmode="group",
        height=500,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption(
        "Fuente: notebook 10 del proyecto (Sprint 3 Fase 5). "
        "LightGBM seleccionado como familia. La selección final por régimen "
        "(pre-shock) se realizó tras el walking-forward del Sprint 4."
    )


def render_historico():
    """Análisis histórico con selector de fecha manual."""
    st.subheader("Consulta histórica del Ministerio")
    
    st.markdown("""
    Selecciona cualquier fecha del histórico para consultar los precios de carburantes 
    publicados ese día por el Ministerio. Si el dato no está en disco, se descargará 
    automáticamente del endpoint histórico.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fecha_seleccionada = st.date_input(
            "Selecciona una fecha",
            value=date(2026, 3, 15),
            min_value=date(2024, 1, 1),
            max_value=date.today() - timedelta(days=1),
            help="Cualquier fecha entre 1 enero 2024 y ayer",
            key="fecha_historico_p4",
        )
    
    with col2:
        carburante_hist = st.selectbox(
            "Carburante",
            options=["Gasóleo A", "Gasolina 95 E5"],
            index=0,
            key="carburante_historico_p4",)
    
    if st.button("🔍 Consultar histórico", type="primary", key="btn_consultar_historico"):
        with st.spinner(f"Buscando datos del {fecha_seleccionada}..."):
            try:
                df_dia = _cargar_o_descargar_dia(fecha_seleccionada, carburante_hist)
            except Exception as e:
                st.error(f"❌ Error al cargar datos: {str(e)}")
                return
        
        if df_dia is None or df_dia.empty:
            st.error(
                f"❌ No se han podido obtener datos para {fecha_seleccionada.strftime('%d/%m/%Y')}. "
                f"Es posible que el Ministerio aún no haya publicado este día en su endpoint histórico "
                f"(suele tardar 1-2 días en consolidarlo). Prueba con una fecha anterior.")
            return
        
        # KPIs del día
        st.subheader(f"📊 Resumen del {fecha_seleccionada.strftime('%d/%m/%Y')}")
        
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        
        with col_k1:
            st.metric(
                "⛽ Estaciones",
                f"{len(df_dia):,}",
            )
        with col_k2:
            st.metric(
                "💰 Precio medio",
                f"{df_dia['precio'].mean():.3f} €/L",
            )
        with col_k3:
            st.metric(
                "📉 Precio mínimo",
                f"{df_dia['precio'].min():.3f} €/L",
            )
        with col_k4:
            st.metric(
                "📈 Precio máximo",
                f"{df_dia['precio'].max():.3f} €/L",
            )
        
        st.divider()
        
        # Top 10 más baratas
        st.markdown("**🏆 Top 10 estaciones más baratas**")
        df_top = df_dia.nsmallest(10, "precio")[
            ["Rotulo_normalizado", "Provincia", "precio", "Localidad"]
        ].copy()
        df_top.columns = ["Marca", "Provincia", "Precio (€/L)", "Localidad"]
        df_top["Precio (€/L)"] = df_top["Precio (€/L)"].round(3)
        df_top.index = range(1, len(df_top) + 1)
        st.dataframe(df_top, use_container_width=True)
        
        # Distribución por provincia
        st.markdown("**📍 Distribución por provincia**")
        df_provincia = df_dia.groupby("Provincia")["precio"].agg(["mean", "count"]).reset_index()
        df_provincia.columns = ["Provincia", "Precio medio", "Estaciones"]
        df_provincia = df_provincia.sort_values("Precio medio").head(15)
        
        fig_prov = go.Figure()
        fig_prov.add_trace(go.Bar(
            x=df_provincia["Precio medio"],
            y=df_provincia["Provincia"],
            orientation="h",
            marker_color=COLOR_PRIMARIO,
            text=df_provincia["Precio medio"].apply(lambda x: f"{x:.3f} €/L"),
            textposition="auto",
        ))
        fig_prov.update_layout(
            title=f"15 provincias más baratas - {carburante_hist}",
            xaxis_title="Precio medio (€/L)",
            yaxis=dict(autorange="reversed"),
            height=500,
        )
        st.plotly_chart(fig_prov, use_container_width=True)


@st.cache_data(ttl=3600)
def _cargar_o_descargar_dia(fecha_seleccionada, carburante):
    """Carga el día desde disco o lo descarga del Ministerio si no existe.
    
    Estrategia de búsqueda:
    1. data/validacion con formato ISO (precios_2026-06-25.json)
    2. data/validacion con formato compacto (precios_20260625.json)
    3. data/raw con formato ISO (descarga histórica del Sprint 2)
    4. Si no existe en ningún sitio, descargar del Ministerio.
    
    Returns:
        DataFrame con datos del día filtrado por carburante, o None si falla.
    """
    from src.consolidacion import cargar_y_limpiar_json
    from src.descarga import descargar_rango
    
    fecha_iso = fecha_seleccionada.isoformat()
    fecha_compacta = fecha_seleccionada.strftime("%Y%m%d")
    
    # Búsqueda en orden de preferencia (solo aceptamos ficheros > 1 MB)
    rutas_buscar = [
        CARPETA_VALIDACION / f"precios_{fecha_iso}.json",
        CARPETA_VALIDACION / f"precios_{fecha_compacta}.json",
        RAIZ_PROYECTO / "data" / "raw" / f"precios_{fecha_iso}.json",
        RAIZ_PROYECTO / "data" / "raw" / f"precios_{fecha_compacta}.json",
    ]
    
    TAMANO_MIN_MB = 1.0  # 1 MB = JSON válido del Ministerio (típico 13 MB)
    
    ruta_a_usar = None
    for r in rutas_buscar:
        if r.exists() and r.stat().st_size > TAMANO_MIN_MB * 1024 * 1024:
            ruta_a_usar = r
            break
    
    # Si no existe, descargar del histórico
    if ruta_a_usar is None:
        logger = logging.getLogger("historico_dashboard")
        ruta_log = RAIZ_PROYECTO / "logs" / "descarga_historico_dashboard.log"
        ruta_log.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            resultado = descargar_rango(
                fecha_inicio=fecha_seleccionada,
                fecha_fin=fecha_seleccionada,
                carpeta_raw=CARPETA_VALIDACION,
                ruta_log=ruta_log,
            )
            if resultado.get("exitosas", 0) == 0:
                return None
        except Exception:
            return None
        
        # Buscar de nuevo tras la descarga
        ruta_a_usar = CARPETA_VALIDACION / f"precios_{fecha_iso}.json"
        if not ruta_a_usar.exists() or ruta_a_usar.stat().st_size < TAMANO_MIN_MB * 1024 * 1024:
            return None
    
    # Cargar y filtrar
    try:
        df = cargar_y_limpiar_json(ruta_a_usar, pd.Timestamp(fecha_seleccionada))
        if df is None or df.empty:
            return None
        
        col_precio = "Precio Gasoleo A" if carburante == "Gasóleo A" else "Precio Gasolina 95 E5"
        df_filtrado = df.dropna(subset=[col_precio]).copy()
        df_filtrado["precio"] = df_filtrado[col_precio]
        
        return df_filtrado
    except Exception:
        return None
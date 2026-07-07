"""
RepostaPro — Dashboard interactivo
TFM Data Science and Business Analytics
Víctor González Martín 

Ejecutar: streamlit run streamlit_app.py
"""

import sys
from pathlib import Path
from datetime import date

# Añadir raíz del proyecto al sys.path para importar src/
RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import streamlit as st

# Imports de las pestañas
from src.dashboard import (
    pestana_1_repostaje,
    pestana_2_ruta,
    pestana_3_flota,
    pestana_4_insights,
)


# ─── CONFIGURACIÓN DE PÁGINA ────────────────────────────────────────────────
st.set_page_config(
    page_title="RepostaPro — Optimización de repostaje",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── ESTILOS CUSTOM ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Banner RepostaPro */
    .banner-repostapro {
        background: linear-gradient(135deg, #DC3545 0%, #8B1A1A 100%);
        padding: 1.8rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(220, 53, 69, 0.25);
    }
    .banner-title {
        color: #FFFFFF;
        font-size: 2.6rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .banner-subtitle {
        color: rgba(255, 255, 255, 0.92);
        font-size: 1.05rem;
        margin: 0.3rem 0 0 0;
        font-weight: 400;
    }
    .banner-author {
        color: rgba(255, 255, 255, 0.78);
        font-size: 0.85rem;
        margin: 0.6rem 0 0 0;
        font-weight: 300;
    }
    
    /* Sidebar más compacto */
    section[data-testid="stSidebar"] .stMarkdown {
        font-size: 0.92rem;
    }
    section[data-testid="stSidebar"] h3 {
        font-size: 1rem;
        margin-top: 0.8rem;
        margin-bottom: 0.4rem;
        color: #DC3545;
    }
    
    /* Tabs principales */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 18px;
        font-size: 0.98rem;
    }
    
    /* Metric KPIs */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ─── BANNER PRINCIPAL ───────────────────────────────────────────────────────
st.markdown("""
<div class="banner-repostapro">
    <p class="banner-title"> RepostaPro</p>
    <p class="banner-subtitle">Sistema inteligente de optimización del repostaje para flotas comerciales</p>
    <p class="banner-author">Víctor González Martín · TFM Data Science and Business Analytics</p>
</div>
""", unsafe_allow_html=True)


# ─── SIDEBAR COMPACTO ───────────────────────────────────────────────────────
with st.sidebar:
    # 1. Botón de actualización (arriba del todo, principal)
    st.markdown("### Actualización")
    
    if st.button(
        "Actualizar precios del Ministerio",
        type="primary",
        use_container_width=True,
    ):
        from src.dashboard.data_loader import actualizar_datos_desde_ministerio
        
        progress_bar = st.progress(0, text="Iniciando...")
        
        def callback(texto, fraccion):
            progress_bar.progress(fraccion, text=texto)
        
        try:
            resultado = actualizar_datos_desde_ministerio(progress_callback=callback)
            progress_bar.empty()
            
            st.success(f"{resultado['mensaje']}")
            
            if resultado.get("descarga_vivo_exitosa"):
                st.caption(
                    f"📡 Datos del Ministerio actualizados a: "
                    f"**{resultado['fecha_actualizacion_ministerio']}**"
                )
                st.caption(f"⛽ {resultado['n_estaciones_vivo']:,} estaciones activas")
            
            st.info("Recarga la página (F5) para ver los datos actualizados.")
        except Exception as e:
            progress_bar.empty()
            st.error(f"Error: {e}")
    
    st.divider()
    
    # 2. Acerca del sistema (compacto)
    st.markdown("### Acerca del sistema")
    st.markdown(
        "**RepostaPro** es un sistema de predicción de precios de carburantes "
        "y optimización de repostaje desarrollado como TFM del "
        "**Máster en Data Science and Business Analytics** de "
        "IMF Smart Education y UCAV."
    )
    
    st.divider()
    
    # 3. Datos del sistema (simplificado)
    st.markdown("### Datos del sistema")
    st.markdown(
        "- **Modelo:** LightGBM gradient boosting\n"
        "- **Histórico:** ene 2024 → actualidad\n"
        "- **Estaciones cubiertas:** ~11.500\n"
        "- **Fuente:** Ministerio de Transición Ecológica"
    )


# ─── PESTAÑAS PRINCIPALES ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Repostaje óptimo",
    "Planificador de rutas",
    "Análisis económico",
    "Insights y métricas",
])

with tab1:
    pestana_1_repostaje.render()

with tab2:
    pestana_2_ruta.render()

with tab3:
    pestana_3_flota.render()

with tab4:
    pestana_4_insights.render()
"""
Funciones de utilidad compartidas entre las pestañas del dashboard.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go

def _memoize_simple(func):
    """Cache simple en memoria para una sola sesión."""
    cache = {}
    def wrapper(*args, **kwargs):
        clave = (args, tuple(sorted(kwargs.items())))
        if clave not in cache:
            cache[clave] = func(*args, **kwargs)
        return cache[clave]
    return wrapper

COLOR_PRIMARIO = "#DC3545"      # rojo 
COLOR_SECUNDARIO = "#E0A458"    # dorado
COLOR_ACENTO = "#4A9D5C"        # verde 
COLOR_ALERTA = "#F0F2F6"        # blanco
COLOR_FONDO = "#0E1117"         # fondo oscuro
COLOR_FONDO_SECUNDARIO = "#1A1D24"  # contenedores
COLOR_TEXTO = "#F0F2F6"         # texto principal
COLOR_TEXTO_SECUNDARIO = "#B0B8C5"  # texto secundario

def crear_mapa_pydeck(df_estaciones, lat_centro, lon_centro, zoom=11,
                       columna_color="precio_real", radio_marcador=120):
    """Crea un mapa PyDeck con marcadores de estaciones coloreados por precio.
    
    NO requiere Mapbox: usa el mapa base Carto que viene integrado en Streamlit.
    
    Args:
        df_estaciones: DataFrame con columnas latitud, longitud, marca, precio_real, etc.
        lat_centro, lon_centro: coordenadas del punto central del mapa.
        zoom: nivel de zoom inicial (1=mundo, 15=barrio).
        columna_color: columna a usar para colorear los marcadores.
        radio_marcador: tamaño del marcador en metros.
    
    Returns:
        pdk.Deck object listo para st.pydeck_chart().
    """
    # Preparar datos para PyDeck
    df = df_estaciones.copy()
    df = df.dropna(subset=["latitud", "longitud", columna_color])
    
    if len(df) == 0:
        # Mapa vacío centrado en el punto
        view_state = pdk.ViewState(
            latitude=lat_centro,
            longitude=lon_centro,
            zoom=zoom,
            pitch=0,
        )
        return pdk.Deck(
            layers=[],
            initial_view_state=view_state,
            map_style="light",
        )
    
    # Normalizar precios para escala de color (0 = más barato → verde, 1 = más caro → rojo)
    precio_min = df[columna_color].min()
    precio_max = df[columna_color].max()
    rango = precio_max - precio_min if precio_max > precio_min else 1
    df["precio_normalizado"] = (df[columna_color] - precio_min) / rango
    
    # Color RGB interpolado verde → amarillo → rojo
    df["color_r"] = (df["precio_normalizado"] * 220).astype(int)
    df["color_g"] = ((1 - df["precio_normalizado"]) * 200 + 40).astype(int)
    df["color_b"] = 50
    
    # Capa de marcadores
    capa_estaciones = pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position=["longitud", "latitud"],
        get_radius=radio_marcador,
        get_fill_color=["color_r", "color_g", "color_b", 180],
        pickable=True,
        auto_highlight=True,
        stroked=True,
        get_line_color=[255, 255, 255, 200],
        line_width_min_pixels=1,
    )
    
    # Tooltip al pasar el ratón
    tooltip = {
        "html": (
            "<b>{marca}</b><br/>"
            "<span style='font-size: 1.1em;'>{precio_real} €/L</span><br/>"
            "Provincia: {provincia}<br/>"
            "ID: {IDEESS}"
        ),
        "style": {
            "backgroundColor": "rgba(26, 79, 139, 0.95)",
            "color": "white",
            "padding": "8px 12px",
            "borderRadius": "6px",
            "fontFamily": "Arial, sans-serif",
        },
    }
    
    # Vista inicial
    view_state = pdk.ViewState(
        latitude=lat_centro,
        longitude=lon_centro,
        zoom=zoom,
        pitch=0,
    )
    
    return pdk.Deck(
        layers=[capa_estaciones],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="light",
    )


def crear_mapa_pydeck_con_radio(df_estaciones, lat_centro, lon_centro, radio_km,
                                   zoom=11, columna_color="precio_real"):
    """Mapa PyDeck con un círculo que representa el radio de búsqueda.
    
    Args:
        igual que crear_mapa_pydeck.
        radio_km: radio del círculo en kilómetros.
    """
    df = df_estaciones.copy()
    df = df.dropna(subset=["latitud", "longitud", columna_color])
    
    # Normalizar precios para escala de color
    if len(df) > 0:
        precio_min = df[columna_color].min()
        precio_max = df[columna_color].max()
        rango = precio_max - precio_min if precio_max > precio_min else 1
        df["precio_normalizado"] = (df[columna_color] - precio_min) / rango
        df["color_r"] = (df["precio_normalizado"] * 220).astype(int)
        df["color_g"] = ((1 - df["precio_normalizado"]) * 200 + 40).astype(int)
        df["color_b"] = 50
    
    capas = []
    
    # Círculo del radio (semitransparente)
    df_circulo = pd.DataFrame([{
        "lat": lat_centro,
        "lon": lon_centro,
        "radio_m": radio_km * 1000,
    }])
    capa_radio = pdk.Layer(
        "ScatterplotLayer",
        df_circulo,
        get_position=["lon", "lat"],
        get_radius="radio_m",
        get_fill_color=[26, 79, 139, 30],
        get_line_color=[26, 79, 139, 180],
        line_width_min_pixels=2,
        stroked=True,
        filled=True,
    )
    capas.append(capa_radio)
    
    # Marcador del centro
    capa_centro = pdk.Layer(
        "ScatterplotLayer",
        pd.DataFrame([{"lat": lat_centro, "lon": lon_centro}]),
        get_position=["lon", "lat"],
        get_radius=200,
        get_fill_color=[26, 79, 139, 255],
        get_line_color=[255, 255, 255, 255],
        line_width_min_pixels=2,
        stroked=True,
    )
    capas.append(capa_centro)
    
    # Marcadores de estaciones
    if len(df) > 0:
        capa_estaciones = pdk.Layer(
            "ScatterplotLayer",
            df,
            get_position=["longitud", "latitud"],
            get_radius=120,
            get_fill_color=["color_r", "color_g", "color_b", 180],
            pickable=True,
            auto_highlight=True,
            stroked=True,
            get_line_color=[255, 255, 255, 200],
            line_width_min_pixels=1,
        )
        capas.append(capa_estaciones)
    
    tooltip = {
        "html": (
            "<b>{marca}</b><br/>"
            "<span style='font-size: 1.1em;'>{precio_real} €/L</span><br/>"
            "Provincia: {provincia}"
        ),
        "style": {
            "backgroundColor": "rgba(26, 79, 139, 0.95)",
            "color": "white",
            "padding": "8px 12px",
            "borderRadius": "6px",
        },
    }
    
    view_state = pdk.ViewState(
        latitude=lat_centro,
        longitude=lon_centro,
        zoom=zoom,
        pitch=0,
    )
    
    return pdk.Deck(
        layers=capas,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="light",
    )


def formatear_euros(valor, decimales=2):
    """Formatea un número como €."""
    return f"{valor:,.{decimales}f} €".replace(",", " ")


def formatear_precio_litro(valor):
    """Formatea un precio en €/L."""
    return f"{valor:.3f} €/L"


def crear_grafico_top_estaciones(df_top, columna_precio="precio_real",
                                    columna_marca="marca"):
    """Gráfico de barras con las top estaciones más baratas."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_top[columna_precio],
        y=df_top[columna_marca],
        orientation="h",
        marker=dict(
            color=df_top[columna_precio],
            colorscale=[[0, COLOR_ACENTO], [1, COLOR_ALERTA]],
            showscale=False,
        ),
        text=df_top[columna_precio].apply(formatear_precio_litro),
        textposition="auto",
    ))
    fig.update_layout(
        title="Top estaciones por precio (más bajo arriba)",
        xaxis_title="Precio (€/L)",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
        height=400,
        margin=dict(l=120, r=20, t=50, b=40),
        showlegend=False,
    )
    return fig 
# ─── GEOCODING ──────────────────────────────────────────────────────────────

@_memoize_simple
def geocodificar_direccion(direccion):
    """Convierte una dirección de texto en coordenadas (latitud, longitud).
    
    Usa Nominatim de OpenStreetMap (gratis, sin token).
    Cacheado en sesión para evitar peticiones repetidas.
    
    Args:
        direccion: dirección en texto, ej: "Plaza Mayor, Madrid".
    
    Returns:
        Tupla (latitud, longitud) o None si no se encuentra.
    """
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    
    try:
        geocoder = Nominatim(user_agent="repostapro-tfm-2026", timeout=5)
        resultado = geocoder.geocode(direccion)
        if resultado is None:
            return None
        return (resultado.latitude, resultado.longitude)
    except (GeocoderTimedOut, GeocoderServiceError):
        return None
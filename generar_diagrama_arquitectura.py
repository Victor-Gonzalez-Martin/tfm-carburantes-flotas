"""
Genera el diagrama de arquitectura funcional del sistema RepostaPro
replicando el estilo vertical con cinco pasos numerados en rojo corporativo.
Guarda el resultado en outputs/figures/arquitectura_repostapro.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from pathlib import Path

# ------------------------------------------------------------------
# Paleta corporativa
# ------------------------------------------------------------------
ROJO_CORP = "#B91C1C"
GRIS_TITULO = "#1F2937"
GRIS_SUBTITULO = "#6B7280"
GRIS_BORDE = "#D1D5DB"
GRIS_FLECHA = "#9CA3AF"
GRIS_PIE = "#6B7280"
GRIS_LINEA = "#E5E7EB"

# ------------------------------------------------------------------
# Contenido de los cinco pasos
# ------------------------------------------------------------------
PASOS = [
    {
        "titulo": "INGESTA DE DATOS",
        "subtitulo": "API REST del Ministerio para la Transición Ecológica",
        "bullets": [
            "Endpoint histórico por fecha",
            "Endpoint en tiempo real",
            "LegacyTLSAdapter sobre OpenSSL 3.0.20",
        ],
    },
    {
        "titulo": "PROCESAMIENTO Y ALMACENAMIENTO",
        "subtitulo": "Limpieza, normalización y persistencia",
        "bullets": [
            "Normalización de marcas (Rotulo_normalizado)",
            "Ingeniería de 18 features anti-leakage",
            "Parquet particionado por año (~11 GB)",
        ],
    },
    {
        "titulo": "MODELADO PREDICTIVO",
        "subtitulo": "Tres paradigmas complementarios",
        "bullets": [
            "Baseline MM7 (media móvil 7 días)",
            "Prophet (descomposición aditiva)",
            "LightGBM (gradient boosting)  \u2014  modelo de producción",
        ],
    },
    {
        "titulo": "OPTIMIZACIÓN Y RECOMENDACIÓN",
        "subtitulo": "Predicción + coste de desvío + restricciones operativas",
        "bullets": [
            "Selección de estación óptima por trayecto",
            "Restricción de autonomía, radio y waypoints",
            "Análisis económico sobre tres perfiles de flota",
        ],
    },
    {
        "titulo": "PRESENTACIÓN",
        "subtitulo": "Dashboard interactivo accesible al usuario final",
        "bullets": [
            "Streamlit + PyDeck (4 pestañas)",
            "Descarga híbrida histórico + tiempo real",
            "Predicción recursiva multi-día (D+1 y D+2)",
        ],
    },
]

# ------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 14))
ax.set_xlim(0, 11)
ax.set_ylim(0, 14)
ax.set_facecolor("white")
ax.axis("off")

# Título principal
ax.text(5.5, 13.5, "ARQUITECTURA FUNCIONAL DE RepostaPro",
        ha="center", va="center",
        fontsize=17, fontweight="bold", color=ROJO_CORP)

# Subtítulo
ax.text(5.5, 12.95, "Flujo de datos: de la fuente pública a la decisión operativa",
        ha="center", va="center",
        fontsize=10.5, style="italic", color=GRIS_SUBTITULO)

# ------------------------------------------------------------------
# Cajas: dimensiones
# ------------------------------------------------------------------
CAJA_X = 0.6
CAJA_ANCHO = 9.8
CAJA_ALTO = 1.75
GAP = 0.35

# Bloque del número rojo (a la izquierda)
NUM_ANCHO = 1.35

# Y de la primera caja (parte superior)
y_top = 12.15

for i, paso in enumerate(PASOS):
    y_bot = y_top - CAJA_ALTO

    # Caja completa (borde gris claro, fondo blanco)
    caja = FancyBboxPatch(
        (CAJA_X, y_bot),
        CAJA_ANCHO, CAJA_ALTO,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2, edgecolor=GRIS_BORDE, facecolor="white"
    )
    ax.add_patch(caja)

    # Bloque numerado rojo a la izquierda (rectángulo)
    num_box = Rectangle(
        (CAJA_X, y_bot),
        NUM_ANCHO, CAJA_ALTO,
        facecolor=ROJO_CORP, edgecolor="none", zorder=2
    )
    ax.add_patch(num_box)

    # Número blanco grande
    ax.text(
        CAJA_X + NUM_ANCHO / 2, y_bot + CAJA_ALTO / 2,
        str(i + 1),
        ha="center", va="center",
        fontsize=36, fontweight="bold", color="white", zorder=3
    )

    # Contenido de texto a la derecha
    texto_x = CAJA_X + NUM_ANCHO + 0.3

    # Título
    ax.text(
        texto_x, y_top - 0.35,
        paso["titulo"],
        ha="left", va="center",
        fontsize=12.5, fontweight="bold", color=GRIS_TITULO
    )

    # Subtítulo (cursiva)
    ax.text(
        texto_x, y_top - 0.65,
        paso["subtitulo"],
        ha="left", va="center",
        fontsize=9.8, style="italic", color=GRIS_SUBTITULO
    )

    # Línea separadora fina
    linea_y = y_top - 0.85
    ax.plot(
        [texto_x, CAJA_X + CAJA_ANCHO - 0.2],
        [linea_y, linea_y],
        color=GRIS_LINEA, linewidth=0.8
    )

    # Bullets
    for j, bullet in enumerate(paso["bullets"]):
        by = linea_y - 0.22 - j * 0.27
        ax.text(
            texto_x + 0.05, by,
            f"\u2022  {bullet}",
            ha="left", va="center",
            fontsize=9.8, color=GRIS_TITULO
        )

    # Flecha vertical hacia la siguiente caja
    if i < len(PASOS) - 1:
        flecha_top = y_bot - 0.03
        flecha_bot = y_bot - GAP + 0.02
        arrow = FancyArrowPatch(
            (5.5, flecha_top), (5.5, flecha_bot),
            arrowstyle="-|>", mutation_scale=18,
            linewidth=1.8, color=GRIS_FLECHA
        )
        ax.add_patch(arrow)

    # Avanzar al siguiente bloque
    y_top = y_bot - GAP

# Pie de página
ax.text(
    5.5, y_top - 0.4,
    "Cada capa tiene responsabilidades acotadas y puede evolucionar de forma independiente.",
    ha="center", va="center",
    fontsize=9.5, style="italic", color=GRIS_PIE
)

# ------------------------------------------------------------------
# Guardado
# ------------------------------------------------------------------
output_dir = Path("outputs/figures")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "arquitectura_repostapro.png"

plt.tight_layout()
plt.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
plt.close()

print(f"Diagrama guardado en: {output_path.resolve()}")
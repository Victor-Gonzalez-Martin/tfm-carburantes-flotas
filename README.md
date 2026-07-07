# RepostaPro — Optimización del repostaje en flotas comerciales

Sistema de predicción de precios de carburantes y optimización del repostaje
para flotas comerciales españolas, desarrollado como Trabajo Fin de Máster del
Máster en Data Science y Business Analytics (IMF Smart Education + UCAV).

**Autor:** Víctor González Martín · **Tutor:** Juan Manuel Moreno Lamparero

---

## Qué hace

RepostaPro aprende de los precios históricos publicados por el Ministerio para
la Transición Ecológica, predice los precios de mañana de cada gasolinera y
recomienda a cada flota dónde y cuándo repostar.

## Resultados clave

| Métrica | Valor |
|---|---|
| Histórico analizado | 895 días (1 ene 2024 – 14 jun 2026) |
| Estaciones cubiertas | 11.475 |
| Error de predicción (MAE, Gasóleo A) | 2,09 cts/L |
| Error de predicción (MAE, Gasolina 95) | 0,67 cts/L |
| Ahorro anual proyectado (3 flotas, 53 vehículos) | 300.356 € (14,9 %) |

## Instalación y uso

```bash
git clone https://github.com/Victor-Gonzalez-Martin/tfm-carburantes-flotas.git
cd tfm-carburantes-flotas
conda env create -f environment.yml
conda activate repostapro
streamlit run streamlit_app.py
```

El dashboard arranca en `http://localhost:8501` con el snapshot de datos de
validación incluido en el repositorio.

## Reconstrucción del histórico completo (opcional)

El histórico completo (~11 GB en Parquet particionado por año) no se incluye
en el repositorio por tamaño. Para reconstruirlo desde la API oficial:

```bash
python src/descarga.py --fecha-inicio 2024-01-01 --fecha-fin 2026-06-14
```

Duración aproximada: 2 horas con conexión estable.

## Estructura del repositorio
├── data/validacion/     Snapshot de precios de la ventana de validación (jun-jul 2026)
├── notebooks/           Análisis exploratorio, modelado y análisis económico
├── src/                 Módulos Python (descarga, features, modelos)
│ └── dashboard/       Pestañas y utilidades del dashboard
├── outputs/             Modelos entrenados, predicciones y figuras
├── streamlit_app.py     Dashboard interactivo (punto de entrada)
└── environment.yml      Entorno conda del proyecto

## Fuente de datos

[API REST de Precios de Carburantes — Ministerio para la Transición Ecológica y el Reto Demográfico](https://sedeaplicaciones.minetur.gob.es/ServiciosRESTCarburantes/PreciosCarburantes/)
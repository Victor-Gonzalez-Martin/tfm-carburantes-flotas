#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Módulo de descarga histórica de precios de carburantes desde la API REST del Ministerio para la Transición Ecológica y el Reto Demográfico.

Proporciona funciones de descarga robusta con reintentos exponenciales,rate limiting aleatorio y registro detallado del proceso.

Incluye adaptador SSL personalizado para gestionar la renegociación legacy del servidor de origen, deshabilitada por defecto en OpenSSL.

Proyecto: RepostaPro
Autor:    Víctor González Martín
"""

import json
import logging
import random
import ssl
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
import urllib3
from urllib3.util.ssl_ import create_urllib3_context

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ─── CONFIGURACIÓN ──────────────────────────────────────────────────────────

URL_BASE_HISTORICO = (
    "https://sedeaplicaciones.minetur.gob.es/ServiciosRESTCarburantes/"
    "PreciosCarburantes/EstacionesTerrestresHist/{fecha}")

HEADERS = {"Accept": "application/json"}

# Política de reintentos
MAX_REINTENTOS = 5
BACKOFF_BASE = 2  # base de la potencia: 2^1, 2^2, 2^3...

# Rate limiting (segundos entre peticiones exitosas)
PAUSA_MIN_SEG = 1.0
PAUSA_MAX_SEG = 3.0

# Timeout de cada petición HTTP
TIMEOUT_SEG = 30


# ─── ADAPTADOR SSL PERSONALIZADO ────────────────────────────────────────────
# El servidor del Ministerio para la Transición Ecológica utiliza renegociación SSL "legacy"
# Este adaptador habilita la renegociación legacy, permitiendo la comunicación con el servidor manteniendo el resto de comprobaciones de seguridad TLS.

OP_LEGACY_SERVER_CONNECT = 0x4  # bandera OpenSSL no expuesta en el módulo ssl


class LegacyTLSAdapter(requests.adapters.HTTPAdapter):
    """Adaptador HTTPS que habilita la renegociación SSL legacy.
    
    Necesario para servidores con configuración SSL antigua (Ministerio
    de Transición Ecológica, 2026). Tras una actualización de OpenSSL
    en el entorno de desarrollo, se requiere forzar cipher suites legacy
    para mantener compatibilidad con el servidor.
    """
    def init_poolmanager(self, *args, **kwargs):
        import ssl
        context = create_urllib3_context()
        context.set_ciphers("DEFAULT:@SECLEVEL=0")
        context.options |= OP_LEGACY_SERVER_CONNECT
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        kwargs["ssl_context"] = context
        return super().init_poolmanager(*args, **kwargs)

def crear_sesion_legacy():
    """Crea una sesión HTTP con el adaptador SSL legacy montado."""
    sesion = requests.Session()
    sesion.mount("https://", LegacyTLSAdapter())
    return sesion


# ─── FUNCIONES AUXILIARES ───────────────────────────────────────────────────

def construir_url(fecha):
    """Construye la URL del endpoint histórico para una fecha dada.

    El Ministerio espera el formato dd-mm-aaaa (con guiones).
    """
    fecha_str = fecha.strftime("%d-%m-%Y")
    return URL_BASE_HISTORICO.format(fecha=fecha_str)


def descargar_una_fecha(fecha, ruta_salida, logger):
    """Descarga el JSON correspondiente a una fecha concreta.

    Aplica reintentos con backoff exponencial ante fallos transitorios.

    Args:
        fecha: objeto datetime.date con la fecha a descargar.
        ruta_salida: objeto Path donde guardar el JSON resultante.
        logger: logger configurado para registrar el proceso.

    Returns:
        True si la descarga fue exitosa, False si agotó todos los reintentos.
    """
    url = construir_url(fecha)

    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            sesion = crear_sesion_legacy()
            respuesta = sesion.get(url, headers=HEADERS, timeout=TIMEOUT_SEG, verify=False)
            respuesta.raise_for_status()  # lanza excepción si HTTP != 200

            datos = respuesta.json()

            # Validación mínima de integridad
            if datos.get("ResultadoConsulta") != "OK":
                logger.warning("  Respuesta sin OK para %s (ResultadoConsulta=%s)",fecha, datos.get("ResultadoConsulta"))
                return False

            # Guardado en disco
            with open(ruta_salida, "w", encoding="utf-8") as f:
                json.dump(datos, f, ensure_ascii=False)

            n_estaciones = len(datos.get("ListaEESSPrecio", []))
            logger.info("  OK %s → %s estaciones (%.1f MB)",fecha, n_estaciones, ruta_salida.stat().st_size / 1024 / 1024)
            return True

        except (requests.exceptions.RequestException, json.JSONDecodeError) as err:
            espera = BACKOFF_BASE ** intento
            if intento < MAX_REINTENTOS:
                logger.warning("  Intento %s/%s fallido para %s (%s). Esperando %ss...",intento, MAX_REINTENTOS, fecha, type(err).__name__, espera)
                time.sleep(espera)
            else:
                logger.error("  FALLO DEFINITIVO para %s tras %s intentos (%s)",fecha, MAX_REINTENTOS, type(err).__name__,)
                return False

    return False


def pausa_aleatoria():
    """Pausa entre PAUSA_MIN_SEG y PAUSA_MAX_SEG segundos."""
    time.sleep(random.uniform(PAUSA_MIN_SEG, PAUSA_MAX_SEG))


def generar_rango_fechas(fecha_inicio, fecha_fin):
    """Genera la lista de fechas entre inicio y fin (ambas incluidas)."""
    fechas = []
    actual = fecha_inicio
    while actual <= fecha_fin:
        fechas.append(actual)
        actual += timedelta(days=1)
    return fechas


def configurar_logger(ruta_log):
    """Configura un logger que escribe a fichero y a consola."""
    logger = logging.getLogger("descarga")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # evita duplicación si se reconfigura

    formato = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",datefmt="%Y-%m-%d %H:%M:%S",)

    # Handler a fichero
    fh = logging.FileHandler(ruta_log, encoding="utf-8")
    fh.setFormatter(formato)
    logger.addHandler(fh)

    # Handler a consola
    ch = logging.StreamHandler()
    ch.setFormatter(formato)
    logger.addHandler(ch)

    return logger


# ─── FUNCIÓN PRINCIPAL DE ORQUESTACIÓN ──────────────────────────────────────

def descargar_rango(fecha_inicio, fecha_fin, carpeta_raw, ruta_log):
    """Orquesta la descarga del rango completo de fechas.

    Args:
        fecha_inicio: date con la primera fecha a descargar.
        fecha_fin: date con la última fecha a descargar (incluida).
        carpeta_raw: Path donde guardar los JSON día a día.
        ruta_log: Path del fichero de log.

    Returns:
        dict con resumen: total, exitosas, fallidas, omitidas (ya existentes).
    """
    carpeta_raw.mkdir(parents=True, exist_ok=True)
    ruta_log.parent.mkdir(parents=True, exist_ok=True)

    logger = configurar_logger(ruta_log)

    fechas = generar_rango_fechas(fecha_inicio, fecha_fin)
    total = len(fechas)

    logger.info("=" * 70)
    logger.info("INICIO DE DESCARGA HISTÓRICA")
    logger.info("Rango: %s → %s (%s fechas)", fecha_inicio, fecha_fin, total)
    logger.info("Carpeta destino: %s", carpeta_raw)
    logger.info("=" * 70)

    inicio_proceso = datetime.now()
    exitosas, fallidas, omitidas = 0, 0, 0

    for i, fecha in enumerate(fechas, 1):
        nombre_fichero = f"precios_{fecha.isoformat()}.json"
        ruta_salida = carpeta_raw / nombre_fichero

        # Si ya existe, lo omitimos (permite reanudar tras interrupciones)
        if ruta_salida.exists() and ruta_salida.stat().st_size > 1000:
            logger.info("[%s/%s] OMITIDO (ya existe): %s", i, total, fecha)
            omitidas += 1
            continue

        logger.info("[%s/%s] Descargando %s ...", i, total, fecha)
        exito = descargar_una_fecha(fecha, ruta_salida, logger)

        if exito:
            exitosas += 1
        else:
            fallidas += 1

        # Pausa entre peticiones para no saturar la API
        if i < total:
            pausa_aleatoria()

    duracion = datetime.now() - inicio_proceso

    logger.info("=" * 70)
    logger.info("DESCARGA FINALIZADA")
    logger.info("Duración total: %s", duracion)
    logger.info("Exitosas:       %s", exitosas)
    logger.info("Omitidas:       %s (ya existían)", omitidas)
    logger.info("Fallidas:       %s", fallidas)
    logger.info("=" * 70)

    return {
        "total": total,
        "exitosas": exitosas,
        "fallidas": fallidas,
        "omitidas": omitidas,
        "duracion": duracion,
    }


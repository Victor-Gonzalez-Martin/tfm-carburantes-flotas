#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Descarga de precios en tiempo real desde el endpoint sin fecha del Ministerio.

A diferencia del módulo src/descarga.py (Sprint 2) que usa el endpoint histórico
(con 1 día de latencia), este módulo usa el endpoint en vivo que devuelve los
precios del día actual actualizados conforme las estaciones reportan.

Endpoint utilizado:
    https://sedeaplicaciones.minetur.gob.es/ServiciosRESTCarburantes/
    PreciosCarburantes/EstacionesTerrestres/

Devuelve también la fecha/hora exacta de la última actualización en el JSON,
permitiendo al sistema mostrar al usuario la frescura del dato.
"""

import json
import logging
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context


URL_TIEMPO_REAL = (
    "https://sedeaplicaciones.minetur.gob.es/ServiciosRESTCarburantes/"
    "PreciosCarburantes/EstacionesTerrestres/"
)


class LegacyTLSAdapter(HTTPAdapter):
    """Adaptador SSL legacy para servidores con SSL antiguo.
    
    Tras la actualización de OpenSSL 3.0.20+ (jun 2026), el servidor del
    Ministerio requiere forzar cipher suites legacy para el handshake SSL.
    """
    def init_poolmanager(self, *args, **kwargs):
        import ssl
        ctx = create_urllib3_context()
        ctx.set_ciphers("DEFAULT:@SECLEVEL=0")
        ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


def _crear_session():
    """Crea una sesión HTTP con el adaptador SSL legacy."""
    s = requests.Session()
    s.mount("https://", LegacyTLSAdapter())
    return s


def descargar_tiempo_real(ruta_salida, logger=None, max_intentos=5):
    """Descarga los precios en tiempo real del Ministerio.

    Args:
        ruta_salida: Path donde guardar el JSON.
        logger: logger opcional.
        max_intentos: número máximo de reintentos.

    Returns:
        Dict con: {'exito': bool, 'fecha_actualizacion': str, 'n_estaciones': int}
        o {'exito': False} en caso de fallo.
    """
    if logger is None:
        logger = logging.getLogger("descarga_tiempo_real")

    session = _crear_session()

    for intento in range(1, max_intentos + 1):
        try:
            response = session.get(URL_TIEMPO_REAL, timeout=30, verify=False)
            response.raise_for_status()

            data = response.json()
            n_estaciones = len(data.get("ListaEESSPrecio", []))
            fecha_actualizacion = data.get("Fecha", "desconocida")

            # Validar que la respuesta tenga estaciones (no respuesta vacía)
            if n_estaciones == 0:
                logger.warning(
                    f"  Respuesta del Ministerio vacía (0 estaciones, "
                    f"fecha={fecha_actualizacion}). Reintentando...")
                espera = 2 ** intento
                time.sleep(espera)
                continue
            
            ruta_salida.parent.mkdir(parents=True, exist_ok=True)
            with open(ruta_salida, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            tamano_mb = ruta_salida.stat().st_size / 1024 / 1024
            
            # Validar tamaño mínimo: 12-16 MB típico; si <1 MB algo va mal
            if tamano_mb < 1.0:
                ruta_salida.unlink()
                logger.warning(
                    f"  Fichero descargado demasiado pequeño ({tamano_mb:.2f} MB). Reintentando...")
                espera = 2 ** intento
                time.sleep(espera)
                continue
            
            logger.info(
                f"  OK tiempo real ({fecha_actualizacion}) → "
                f"{n_estaciones} estaciones ({tamano_mb:.1f} MB)")
            return {
                "exito": True,
                "fecha_actualizacion": fecha_actualizacion,
                "n_estaciones": n_estaciones,
                "tamano_mb": tamano_mb,}

        except Exception as e:
            espera = 2 ** intento
            logger.warning(
                f"  Intento {intento}/{max_intentos} fallido "
                f"(tiempo real, {type(e).__name__}). Esperando {espera}s...")
            time.sleep(espera)

    logger.error(f"  FALLO DEFINITIVO en tiempo real tras {max_intentos} intentos")
    return {"exito": False}


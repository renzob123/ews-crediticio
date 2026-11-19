"""
Tesis EWS Crediticio - v2
Script 01: Descarga series macroeconómicas del BCRA

Genera: macro_bcra.csv (una fila por mes, separador ;)

Fuentes BCRA:
  panser.txt   - Panorama monetario y financiero
  din1_ser.txt - Dinero y crédito (situación de cartera del sistema)
  tas1_ser.txt - Tasas de interés pasivas (BADLAR)
  tas2_ser.txt - Tasas de interés activas
  API v4.0 de estadísticas monetarias, variable 30 - CER (diario)

Columnas (ver docs/decisiones_metodologicas.md, dudas 2 y 5):
  Las 10 variables macro del modelo
    mora_3plus_privado_pct, sit2_privado_pct, var_mora3plus_3m, var_sit2_3m,
    tasa_consumo_corto_pct, badlar_privados_tna_pct, spread_consumo_badlar_pct,
    var_tc_3m_pct, var_credito_real_3m_pct, pct_personales
  Auxiliares (no van al modelo)
    cer                       CER del último día del mes; 03 lo usa para deflactar
    tipo_cambio_referencia    para var_tc_3m_pct
    credito_privado_mn_miles_pesos, credito_privado_total_miles_pesos,
    prestamos_personales_mn_miles_pesos

Los meses que el BCRA todavía no publicó (o que faltan en la serie) quedan
VACÍOS, no en 0. Las variaciones a 3 meses comparan con el mismo mes calendario
3 meses antes y quedan vacías si falta cualquiera de los dos valores.
Las variaciones de mora (var_mora3plus_3m, var_sit2_3m) se calculan sobre el
sector privado, igual que los niveles que se usan.

El rezago de publicación (usar t o t-1 según qué estaba publicado al momento del
scoring) NO se aplica acá: este archivo describe cada mes calendario y el
rezago lo aplica 03_split.py.

Uso: python 01_descargar_macro_bcra.py
"""

import urllib.request
import ssl
import csv
import json
import math
from datetime import datetime, date

# ── configuración ───────────────────────────────────────────────────────────────
FECHA_DESDE = datetime(2015, 1, 1)   # histórico completo para análisis descriptivo
OUTPUT      = "macro_bcra.csv"

BASE    = "https://www.bcra.gob.ar/archivos/Pdfs/PublicacionesEstadisticas"
FUENTES = {
    "panser": f"{BASE}/panser.txt",
    "din1":   f"{BASE}/din1_ser.txt",
    "tas1":   f"{BASE}/tas1_ser.txt",
    "tas2":   f"{BASE}/tas2_ser.txt",
}
API_CER = ("https://api.bcra.gob.ar/estadisticas/v4.0/monetarias/30"
           "?desde={desde}&hasta={hasta}&limit=1000")

# ── series a extraer ────────────────────────────────────────────────────────────
SERIES = {
    # Situación de cartera - sector privado (din1)
    1109: "sit2_privado_pct",
    1110: "sit3a_privado_pct",
    1111: "sit3b_privado_pct",
    1112: "sit4a_privado_pct",
    1113: "sit4b_privado_pct",
    1114: "sit5_privado_pct",
    # Tipo de cambio y crédito (panser)
    23: "credito_privado_mn_miles_pesos",
    27: "tipo_cambio_referencia",
    # Tasas (tas1 y tas2)
    2302: "badlar_privados_tna_pct",
    1751: "tasa_consumo_corto_pct",
    # Composición de cartera privada MN (din1)
    149: "credito_privado_total_miles_pesos",
    158: "prestamos_personales_mn_miles_pesos",   # incluye tarjetas (def. BCRA)
}

HEADER = [
    "periodo_aaaamm", "fecha_bcra",
    # variables del modelo
    "mora_3plus_privado_pct", "sit2_privado_pct",
    "var_mora3plus_3m", "var_sit2_3m",
    "tasa_consumo_corto_pct", "badlar_privados_tna_pct", "spread_consumo_badlar_pct",
    "var_tc_3m_pct", "var_credito_real_3m_pct", "pct_personales",
    # auxiliares
    "cer", "tipo_cambio_referencia",
    "credito_privado_mn_miles_pesos", "credito_privado_total_miles_pesos",
    "prestamos_personales_mn_miles_pesos",
]


# ── descarga ────────────────────────────────────────────────────────────────────
def descargar(nombre, url, ctx):
    print(f"  Descargando {nombre}...", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
            content = r.read().decode("latin-1")
        print(f"{len(content):,} chars OK")
        return content
    except Exception as e:
        print(f"ERROR: {e}")
        return ""


def parsear_series(content, codigos, fecha_desde):
    """Devuelve {periodo AAAAMM: {code: valor}}; los valores vacíos no se cargan."""
    data = {}
    for line in content.splitlines():
        parts = line.strip().split(";")
        if len(parts) != 3:
            continue
        try:
            code = int(parts[0])
        except ValueError:
            continue
        if code not in codigos:
            continue
        try:
            fecha = datetime.strptime(parts[1].strip(), "%d/%m/%Y")
            valor = float(parts[2].strip())
        except ValueError:
            continue
        if fecha < fecha_desde:
            continue
        data.setdefault(fecha.year * 100 + fecha.month, {})[code] = valor
    return data


def descargar_cer(ctx, desde, hasta):
    """CER del último día disponible de cada mes: {periodo AAAAMM: valor}."""
    print("  Descargando CER (API v4.0)...", end=" ", flush=True)
    ultimo = {}   # periodo → (fecha, valor)
    for anio in range(desde.year, hasta.year + 1):
        url = API_CER.format(desde=f"{anio}-01-01", hasta=f"{anio}-12-31")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
            resp = json.loads(r.read().decode("utf-8"))
        for res in resp.get("results", []):
            for obs in res.get("detalle", []):
                f = datetime.strptime(obs["fecha"], "%Y-%m-%d")
                p = f.year * 100 + f.month
                if p not in ultimo or f > ultimo[p][0]:
                    ultimo[p] = (f, float(obs["valor"]))
    print(f"{len(ultimo)} meses OK")
    return {p: v for p, (_, v) in ultimo.items()}


# ── helpers ─────────────────────────────────────────────────────────────────────
NAN = float("nan")

def menos_meses(periodo, k):
    total = (periodo // 100) * 12 + periodo % 100 - 1 - k
    return (total // 12) * 100 + total % 12 + 1

def ok(*xs):
    return all(x is not None and not math.isnan(x) for x in xs)

def fin_de_mes(periodo):
    y, m = periodo // 100, periodo % 100
    primero_sig = date(y + (m == 12), m % 12 + 1, 1)
    return date.fromordinal(primero_sig.toordinal() - 1).strftime("%d/%m/%Y")

def fmt(x, nd):
    return "" if not ok(x) else round(x, nd)


# ── construcción del CSV ────────────────────────────────────────────────────────
def main():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode    = ssl.CERT_NONE

    print("=" * 55)
    print("Script 01 - Descarga macro BCRA")
    print("=" * 55)

    data = {}   # periodo → {code: valor}
    codigos = set(SERIES)
    for nombre, url in FUENTES.items():
        for p, vals in parsear_series(descargar(nombre, url, ctx), codigos, FECHA_DESDE).items():
            data.setdefault(p, {}).update(vals)
    cer = descargar_cer(ctx, FECHA_DESDE, datetime.now())

    # el CER se conoce por adelantado: no se agregan meses que solo tienen CER
    periodos = sorted(data)
    print(f"\n  Períodos: {len(periodos)} ({periodos[0]} → {periodos[-1]})")

    # valores base por período (NaN si falta). Un 0 exacto en estas series no
    # es un dato real (tasas, porcentajes, stocks del sistema): se toma como faltante.
    def g(p, code):
        v = data.get(p, {}).get(code)
        return NAN if v is None or v == 0 else v

    base = {}
    for p in periodos:
        comps = [g(p, c) for c in (1110, 1111, 1112, 1113, 1114)]
        mora3 = sum(comps) if ok(*comps) else NAN
        cred_mn  = g(p, 23)
        cer_p    = cer.get(p, NAN)
        base[p] = {
            "mora_3plus_privado_pct": mora3,
            "sit2_privado_pct":       g(p, 1109),
            "tasa_consumo_corto_pct": g(p, 1751),
            "badlar_privados_tna_pct": g(p, 2302),
            "tipo_cambio_referencia": g(p, 27),
            "credito_privado_mn_miles_pesos": cred_mn,
            "credito_real": cred_mn / cer_p if ok(cred_mn, cer_p) else NAN,
            "credito_privado_total_miles_pesos": g(p, 149),
            "prestamos_personales_mn_miles_pesos": g(p, 158),
            "cer": cer_p,
        }

    def hace3(p, col):
        b = base.get(menos_meses(p, 3))
        return b[col] if b else NAN

    def dif(a, b):  return a - b if ok(a, b) else NAN
    def var(a, b):  return (a / b - 1) * 100 if ok(a, b) and b != 0 else NAN

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(HEADER)
        for p in periodos:
            b = base[p]
            tot, pers = b["credito_privado_total_miles_pesos"], b["prestamos_personales_mn_miles_pesos"]
            w.writerow([
                p, fin_de_mes(p),
                fmt(b["mora_3plus_privado_pct"], 6),
                fmt(b["sit2_privado_pct"], 6),
                fmt(dif(b["mora_3plus_privado_pct"], hace3(p, "mora_3plus_privado_pct")), 4),
                fmt(dif(b["sit2_privado_pct"], hace3(p, "sit2_privado_pct")), 4),
                fmt(b["tasa_consumo_corto_pct"], 4),
                fmt(b["badlar_privados_tna_pct"], 4),
                fmt(dif(b["tasa_consumo_corto_pct"], b["badlar_privados_tna_pct"]), 4),
                fmt(var(b["tipo_cambio_referencia"], hace3(p, "tipo_cambio_referencia")), 4),
                fmt(var(b["credito_real"], hace3(p, "credito_real")), 4),
                fmt(pers / tot * 100 if ok(pers, tot) else NAN, 4),
                fmt(b["cer"], 6),
                fmt(b["tipo_cambio_referencia"], 4),
                fmt(b["credito_privado_mn_miles_pesos"], 0),
                fmt(tot, 0),
                fmt(pers, 0),
            ])

    print(f"\n  Archivo generado: {OUTPUT}")
    print(f"  Filas: {len(periodos)} | Columnas: {len(HEADER)}")
    print("  Meses faltantes (vacíos) en los últimos 36 meses:")
    for col in ("mora_3plus_privado_pct", "sit2_privado_pct", "tasa_consumo_corto_pct",
                "badlar_privados_tna_pct", "tipo_cambio_referencia",
                "credito_privado_mn_miles_pesos", "prestamos_personales_mn_miles_pesos", "cer"):
        faltan = [p for p in periodos[-36:] if not ok(base[p][col])]
        print(f"    {col:<38} {faltan if faltan else '-'}")
    print("\nListo. Siguiente paso: 02_construir_dataset.py y 03_split.py.")


if __name__ == "__main__":
    main()

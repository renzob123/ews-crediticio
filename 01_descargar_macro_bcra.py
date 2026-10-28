"""
Tesis EWS Crediticio - v2
Script 01: Descarga series macroeconómicas del BCRA

Genera: macro_bcra.csv (una fila por mes, separador ;)

Fuentes BCRA:
  panser.txt   - Panorama monetario y financiero
  din1_ser.txt - Dinero y crédito (situación de cartera del sistema)
  tas1_ser.txt - Tasas de interés pasivas (BADLAR)
  tas2_ser.txt - Tasas de interés activas

Uso: python 01_descargar_macro_bcra.py

Nota: FECHA_DESDE puede extenderse hasta 2015 para gráficos de contexto
en la tesis; el script 02 solo usará los meses que coincidan con el 24DSF.
"""

import urllib.request
import ssl
import csv
from datetime import datetime

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

# ── series a extraer ────────────────────────────────────────────────────────────
SERIES = {
    # Situación de cartera - sistema total (din1)
    1080: "total_financ_sistema_miles_pesos",
    1082: "sit2_sistema_pct",
    1083: "sit3a_sistema_pct",         # sit3 vencida
    1084: "sit3b_sistema_pct",         # sit3 no vencida
    1085: "sit4a_sistema_pct",
    1086: "sit4b_sistema_pct",
    1087: "sit5a_sistema_pct",
    1088: "sit5b_sistema_pct",
    # Situación de cartera - sector privado (din1)
    1107: "total_financ_privado_miles_pesos",
    1109: "sit2_privado_pct",
    1110: "sit3a_privado_pct",
    1111: "sit3b_privado_pct",
    1112: "sit4a_privado_pct",
    1113: "sit4b_privado_pct",
    1114: "sit5_privado_pct",
    # Macro: tipo de cambio, crédito, depósitos (panser)
    1:  "reservas_fin_mes_mmusd",
    15: "base_monetaria_miles_pesos",
    18: "depositos_privado_mn_miles_pesos",
    23: "credito_privado_mn_miles_pesos",
    24: "credito_privado_me_miles_pesos",
    27: "tipo_cambio_referencia",
    # Tasas (tas1 y tas2)
    2300: "badlar_total_tna_pct",
    2302: "badlar_privados_tna_pct",
    1751: "tasa_consumo_corto_pct",
    1752: "tasa_consumo_largo_pct",
    37:   "tasa_adelantos_privado_pct",
    39:   "tasa_hipotecarios_privado_pct",
    # Rentabilidad (din1)
    1150: "resultado_neto_mensual_miles_pesos",
    1151: "intereses_ganados_miles_pesos",
    1157: "intereses_pagados_miles_pesos",
    1208: "n_entidades_financieras",
    # Composición de cartera privada MN (din1)
    149:  "credito_privado_total_miles_pesos",
    152:  "prestamos_adelantos_mn_miles_pesos",
    153:  "prestamos_descuentos_mn_miles_pesos",
    154:  "prestamos_hipotecarios_mn_miles_pesos",
    157:  "prestamos_prendarios_mn_miles_pesos",
    158:  "prestamos_personales_mn_miles_pesos",   # incluye tarjetas (def. BCRA)
}


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
        fecha_str = parts[1].strip()
        try:
            fecha = datetime.strptime(fecha_str, "%d/%m/%Y")
        except ValueError:
            continue
        if fecha < fecha_desde:
            continue
        valor_str = parts[2].strip()
        if fecha_str not in data:
            data[fecha_str] = {}
        try:
            data[fecha_str][code] = float(valor_str) if valor_str else None
        except ValueError:
            data[fecha_str][code] = None
    return data


# ── construcción del CSV ────────────────────────────────────────────────────────
def main():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode    = ssl.CERT_NONE

    print("=" * 55)
    print("Script 01 - Descarga macro BCRA")
    print("=" * 55)

    contenidos = {nombre: descargar(nombre, url, ctx)
                  for nombre, url in FUENTES.items()}

    # unificar en un dict fecha → {code: valor}
    data_total = {}
    codigos = set(SERIES.keys())
    for contenido in contenidos.values():
        parcial = parsear_series(contenido, codigos, FECHA_DESDE)
        for fecha, vals in parcial.items():
            if fecha not in data_total:
                data_total[fecha] = {}
            data_total[fecha].update(vals)

    fechas = sorted(data_total.keys(),
                    key=lambda s: datetime.strptime(s, "%d/%m/%Y"))
    print(f"\n  Períodos disponibles: {len(fechas)}"
          f" ({fechas[0]} → {fechas[-1]})")

    # pre-calcular mora sistémica para calcular variaciones
    mora_3plus = {}
    mora_sit2  = {}
    for f in fechas:
        d = data_total[f]
        g = lambda c: d.get(c) or 0.0
        sit3 = round(g(1083) + g(1084), 6)
        sit4 = round(g(1085) + g(1086), 6)
        sit5 = round(g(1087) + g(1088), 6)
        mora_3plus[f] = round(sit3 + sit4 + sit5, 6)
        mora_sit2[f]  = round(g(1082), 6)

    # columnas del output
    header = [
        "periodo_aaaamm",      # YYYYMM para join con el dataset ML
        "fecha_bcra",          # DD/MM/YYYY original del BCRA
        # Mora sistémica - sistema total
        "sit2_sistema_pct",
        "sit3_sistema_pct",
        "sit4_sistema_pct",
        "sit5_sistema_pct",
        "mora_3plus_sistema_pct",          # sit3+sit4+sit5
        "cartera_irregular_sistema_pct",   # sit2+sit3+sit4+sit5
        # Mora - sector privado
        "sit2_privado_pct",
        "sit3_privado_pct",
        "sit4_privado_pct",
        "sit5_privado_pct",
        "mora_3plus_privado_pct",
        "cartera_irregular_privado_pct",
        # Variaciones de mora sistémica (señales leading)
        "var_mora3plus_1m",
        "var_mora3plus_3m",
        "var_sit2_1m",
        "var_sit2_3m",
        # Macro
        "tipo_cambio_referencia",
        "reservas_fin_mes_mmusd",
        "base_monetaria_miles_pesos",
        "depositos_privado_mn_miles_pesos",
        "credito_privado_mn_miles_pesos",
        "credito_privado_me_miles_pesos",
        # Tasas
        "badlar_total_tna_pct",
        "badlar_privados_tna_pct",
        "tasa_consumo_corto_pct",
        "tasa_consumo_largo_pct",
        "tasa_adelantos_privado_pct",
        "tasa_hipotecarios_privado_pct",
        "spread_consumo_badlar_pct",       # tasa_consumo_corto - badlar_privados
        # Rentabilidad
        "resultado_neto_mensual_miles_pesos",
        "intereses_ganados_miles_pesos",
        "intereses_pagados_miles_pesos",
        "margen_intermediacion_miles_pesos",
        "roa_proxy_pct",
        # Composición de cartera
        "credito_privado_total_miles_pesos",
        "prestamos_adelantos_mn_miles_pesos",
        "prestamos_descuentos_mn_miles_pesos",
        "prestamos_hipotecarios_mn_miles_pesos",
        "prestamos_prendarios_mn_miles_pesos",
        "prestamos_personales_mn_miles_pesos",
        "pct_adelantos",
        "pct_descuentos",
        "pct_hipotecarios",
        "pct_prendarios",
        "pct_personales",
        # Estructura
        "n_entidades_financieras",
        "total_financ_sistema_miles_pesos",
        "total_financ_privado_miles_pesos",
    ]

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)

        for i, fecha in enumerate(fechas):
            d  = data_total[fecha]
            g  = lambda c: d.get(c) or 0.0

            # período AAAAMM para join
            dt = datetime.strptime(fecha, "%d/%m/%Y")
            periodo = dt.year * 100 + dt.month

            # mora sistema
            sit2_sis  = round(g(1082), 6)
            sit3_sis  = round(g(1083) + g(1084), 6)
            sit4_sis  = round(g(1085) + g(1086), 6)
            sit5_sis  = round(g(1087) + g(1088), 6)
            m3p_sis   = round(sit3_sis + sit4_sis + sit5_sis, 6)
            irreg_sis = round(sit2_sis + m3p_sis, 6)

            # mora privado
            sit2_priv  = round(g(1109), 6)
            sit3_priv  = round(g(1110) + g(1111), 6)
            sit4_priv  = round(g(1112) + g(1113), 6)
            sit5_priv  = round(g(1114), 6)
            m3p_priv   = round(sit3_priv + sit4_priv + sit5_priv, 6)
            irreg_priv = round(sit2_priv + m3p_priv, 6)

            # variaciones (vacío si no hay historia suficiente)
            vm3_1m = round(mora_3plus[fecha] - mora_3plus[fechas[i-1]], 4) if i >= 1 else ""
            vm3_3m = round(mora_3plus[fecha] - mora_3plus[fechas[i-3]], 4) if i >= 3 else ""
            vs2_1m = round(mora_sit2[fecha]  - mora_sit2[fechas[i-1]],  4) if i >= 1 else ""
            vs2_3m = round(mora_sit2[fecha]  - mora_sit2[fechas[i-3]],  4) if i >= 3 else ""

            # tasas
            badlar   = g(2302)
            t_corto  = g(1751)
            spread   = round(t_corto - badlar, 4) if badlar and t_corto else ""

            # rentabilidad
            int_gan  = g(1151)
            int_pag  = g(1157)
            margen   = round(int_gan - int_pag, 2)
            cred_tot = g(149)
            roa      = round(g(1150) / cred_tot * 100, 4) if cred_tot else ""

            # composición cartera
            adelantos    = g(152)
            descuentos   = g(153)
            hipotecarios = g(154)
            prendarios   = g(157)
            personales   = g(158)
            pct = lambda x: round(x / cred_tot * 100, 4) if cred_tot else ""

            row = [
                periodo, fecha,
                sit2_sis, sit3_sis, sit4_sis, sit5_sis, m3p_sis, irreg_sis,
                sit2_priv, sit3_priv, sit4_priv, sit5_priv, m3p_priv, irreg_priv,
                vm3_1m, vm3_3m, vs2_1m, vs2_3m,
                round(g(27), 4), round(g(1), 2),
                round(g(15)), round(g(18)), round(g(23)), round(g(24)),
                round(g(2300), 4), round(badlar, 4),
                round(t_corto, 4), round(g(1752), 4),
                round(g(37), 4), round(g(39), 4), spread,
                round(g(1150)), round(int_gan), round(int_pag), margen, roa,
                round(cred_tot), round(adelantos), round(descuentos),
                round(hipotecarios), round(prendarios), round(personales),
                pct(adelantos), pct(descuentos), pct(hipotecarios),
                pct(prendarios), pct(personales),
                int(g(1208)) if g(1208) else "",
                round(g(1080)), round(g(1107)),
            ]
            w.writerow(row)

    print(f"\n  Archivo generado: {OUTPUT}")
    print(f"  Filas: {len(fechas)} | Columnas: {len(header)}")
    print(f"  Período: {fechas[0]} → {fechas[-1]}")
    print("\nListo. Correr 02_construir_dataset.py como siguiente paso.")


if __name__ == "__main__":
    main()

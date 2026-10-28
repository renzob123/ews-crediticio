"""
Tesis EWS Crediticio - v2
Script 02: Construcción del dataset ML desde el 24DSF

Hace dos pasadas sobre el 24DSF y genera directamente el dataset listo
para modelar (una fila por CUIT × período válido).

Filtros aplicados en pasada 1 (sobre los clientes de Santander):
  - Entidad focal: 72 (Santander Argentina)
  - Persona física: prefijo CUIT 20, 23, 24 o 27
  - Mayor de edad estimado: DNI < 48.000.000 (nacidos antes de 2008)
  - Excluye extranjeros: DNI >= 90.000.000

Targets:
  target_a: caída a sit >= 2 en cualquier entidad en los próximos 3 meses
            (condición de entrada: sit = 1 en TODO el sistema en t)
  target_c: caída a sit >= 3 en cualquier entidad en los próximos 12 meses
            (condición de entrada: sit <= 2 en TODO el sistema en t)

Features principales:
  Santander:
    - sit_sant_t, sit_sant_t1, sit_sant_t2, sit_sant_t3
    - monto_sant_t, monto_sant_t3, var_monto_3m_sant
    - meses_consec_normal_sant (rachas sit=1 hacia atrás)
    - judicial_sant_t
    - meses_con_deuda_sant

  Otras entidades (excluye Santander):
    - n_entidades_otras_t        (cuántas otras entidades tiene activas)
    - peor_sit_otras_t           (peor situación en otras entidades)
    - n_ent_sit2_otras_t
    - n_ent_sit3plus_otras_t
    - tiene_judicial_otras_t
    - log_monto_otras_t
    - otras_tiene_banco_publico / privado / digital / financiera / billetera / prov_nofinanciero
    - otras_n_tipos_entidad

  Sistema total (Santander + otras):
    - n_entidades_total_t        (todas las entidades incluyendo Santander)
    - peor_sit_sistema_t         (peor sit en todo el sistema)
    - meses_con_deuda_sistema

  Demografía (derivada del CUIT):
    - sexo_proxy (M/F/I)
    - rango_edad_proxy

  Antigüedad:
    - left_censored (ya estaba en el mes más antiguo del archivo)

Uso: python 02_construir_dataset.py
Tiempo estimado: 30-50 min (dos pasadas sobre ~22GB)
Requiere: pyarrow (pip install pyarrow)
"""

import os
import math
from collections import defaultdict
import pyarrow as pa
import pyarrow.parquet as pq

# ── configuración ───────────────────────────────────────────────────────────────
ARCHIVO_24DSF = "24DSF.txt"     # archivo agosto 2026 (24 meses: sep 2024 → ago 2026)
BASE_PERIODO  = 202608          # AAAAMM del mes índice 0 (más reciente)
ENTIDAD_FOCAL = "72"            # Santander Argentina
HORIZON_A     = 3               # meses a futuro para target_a
HORIZON_C     = 12              # meses a futuro para target_c
DNI_MAYOR_18  = 48_000_000      # DNI < este → mayor de 18 en 2026
DNI_EXTRANJERO = 90_000_000     # DNI >= este → extranjero, se excluye
OUTPUT        = "dataset_ews.parquet"

# ── clasificación de entidades ──────────────────────────────────────────────────
BANCOS_PUBLICOS = {
    11, 14, 20, 29, 44, 45, 65, 83, 86, 93, 94, 97,
    268, 300, 309, 311, 315, 321, 330, 386,
}
BANCOS_DIGITALES = {143, 147, 150, 310, 312, 384}
BANCOS_PRIVADOS  = {
    7, 15, 16, 17, 27, 34, 72, 131, 165, 191, 198,
    247, 254, 259, 266, 269, 277, 281, 285, 299,
    301, 305, 319, 322, 331, 332, 338, 339, 340,
    341, 389, 426, 431, 432, 435, 448,
}
BILLETERAS = {72634, 40011}

def tipo_entidad(cod_str):
    try:
        cod = int(cod_str)
    except ValueError:
        return "prov_nofinanciero"
    if cod in BANCOS_DIGITALES: return "banco_digital"
    if cod in BANCOS_PUBLICOS:  return "banco_publico"
    if cod in BANCOS_PRIVADOS:  return "banco_privado"
    if cod in BILLETERAS:       return "billetera"
    if 44000 <= cod < 46000 or 65000 <= cod < 66000:
        return "financiera"
    return "prov_nofinanciero"


# ── mapeo índice → período AAAAMM ──────────────────────────────────────────────
# idx 0 = más reciente (BASE_PERIODO), idx 23 = más antiguo
def idx_a_periodo(base, idx):
    y, m = base // 100, base % 100
    m -= idx
    while m <= 0:
        m += 12
        y -= 1
    return y * 100 + m

PERIODOS = [idx_a_periodo(BASE_PERIODO, i) for i in range(24)]


# ── parser de línea 24DSF ───────────────────────────────────────────────────────
def parsear(line):
    """
    Estructura: 5 entidad | 2 tipo_id | 11 CUIT | 24 × (2 sit + 12 monto + 1 judicial)
    Total mínimo: 378 chars.
    Retorna (entidad, cuit, meses) donde meses[i] = (sit, monto_miles, judicial)
    i=0 → mes más reciente, i=23 → más antiguo.
    """
    if len(line) < 378:
        return None
    entidad = line[0:5].strip()
    cuit    = line[7:18]
    meses   = []
    for i in range(24):
        p = 18 + i * 15
        s_str = line[p:p+2].strip()
        m_str = line[p+2:p+14].strip().replace(",", ".")
        j_str = line[p+14:p+15].strip()
        try:
            s = int(s_str)   if s_str else 0
            m = float(m_str) if m_str else 0.0
            j = int(j_str)   if j_str else 0
        except ValueError:
            s, m, j = 0, 0.0, 0
        meses.append((s, m, j))
    return entidad, cuit, meses


# ── filtro persona física mayor de edad ────────────────────────────────────────
def es_elegible(cuit):
    if len(cuit) < 11:
        return False
    prefijo = cuit[:2]
    if prefijo not in ("20", "23", "24", "27"):
        return False
    try:
        dni = int(cuit[2:10])
    except ValueError:
        return False
    return dni < DNI_MAYOR_18 and dni < DNI_EXTRANJERO


# ── demografía desde CUIT ───────────────────────────────────────────────────────
def demograficos(cuit):
    prefijo = cuit[:2]
    if prefijo == "20":   sexo = "M"
    elif prefijo == "27": sexo = "F"
    else:                 sexo = "I"

    try:
        dni = int(cuit[2:10])
    except ValueError:
        return sexo, "desconocido"

    if dni >= 38_000_000:   rango = "18_30"
    elif dni >= 24_000_000: rango = "30_50"
    elif dni >= 14_000_000: rango = "50_65"
    else:                   rango = "65_mas"
    return sexo, rango


# ── helpers de acceso a meses ──────────────────────────────────────────────────
def sit(meses, idx):
    return meses[idx][0] if 0 <= idx < len(meses) else 0

def monto(meses, idx):
    return meses[idx][1] if 0 <= idx < len(meses) else 0.0

def jud(meses, idx):
    return meses[idx][2] if 0 <= idx < len(meses) else 0

def meses_consec_normal(meses, desde_idx):
    """Meses consecutivos con sit=1 desde desde_idx hacia el pasado."""
    c, i = 0, desde_idx
    while i < len(meses) and meses[i][0] == 1:
        c += 1
        i += 1
    return c

def meses_con_deuda(meses, desde_idx):
    """Meses con sit > 0 desde desde_idx hacia el pasado."""
    return sum(1 for i in range(desde_idx, len(meses)) if meses[i][0] > 0)


# ── main ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Script 02 - Construcción dataset EWS v2")
    print("=" * 60)
    print(f"  Archivo    : {ARCHIVO_24DSF}")
    print(f"  Base período: {BASE_PERIODO}  ({PERIODOS[23]} → {PERIODOS[0]})")
    print(f"  Target A   : caída a sit>=2 en {HORIZON_A} meses")
    print(f"  Target C   : caída a sit>=3 en {HORIZON_C} meses")
    print()

    if not os.path.exists(ARCHIVO_24DSF):
        print(f"ERROR: no se encuentra {ARCHIVO_24DSF}")
        return

    # ── PASADA 1: identificar CUITs elegibles de Santander ─────────────────────
    print("[1] Pasada 1 — CUITs de Santander con prefiltro...")
    sant = {}   # cuit → lista de 24 tuplas (sit, monto, judicial)
    n = 0
    with open(ARCHIVO_24DSF, encoding="latin-1") as f:
        for line in f:
            n += 1
            if n % 5_000_000 == 0:
                print(f"  {n/1e6:.0f}M líneas | {len(sant):,} CUITs elegibles")
            r = parsear(line.rstrip("\r\n"))
            if not r:
                continue
            entidad, cuit, meses = r
            if entidad != ENTIDAD_FOCAL:
                continue
            if not es_elegible(cuit):
                continue
            if any(m[0] > 0 for m in meses):
                sant[cuit] = meses

    print(f"\n  Total líneas   : {n:,}")
    print(f"  CUITs elegibles: {len(sant):,}")

    # ── PASADA 2: acumular otras entidades para esos CUITs ─────────────────────
    # Para cada CUIT × mes guardamos un acumulador con lo necesario para
    # calcular las features de otras entidades al final, sin guardar los
    # meses crudos de cada entidad.
    #
    # otras[cuit][idx] = {
    #   "sits":   [sit_ent1, sit_ent2, ...],   ← para peor_sit, n_sit2, n_sit3+
    #   "montos": [monto_ent1, ...],
    #   "juds":   [jud_ent1, ...],
    #   "tipos":  set(tipo_ent1, tipo_ent2, ...) ← para dummies de tipo
    # }
    print(f"\n[2] Pasada 2 — acumulando otras entidades para {len(sant):,} CUITs...")
    otras = {cuit: [{"sits": [], "montos": [], "juds": [], "tipos": set()}
                    for _ in range(24)]
             for cuit in sant}
    n = 0
    with open(ARCHIVO_24DSF, encoding="latin-1") as f:
        for line in f:
            n += 1
            if n % 5_000_000 == 0:
                print(f"  {n/1e6:.0f}M líneas")
            r = parsear(line.rstrip("\r\n"))
            if not r:
                continue
            entidad, cuit, meses = r
            if entidad == ENTIDAD_FOCAL or cuit not in sant:
                continue
            t = tipo_entidad(entidad)
            for idx in range(24):
                s = meses[idx][0]
                if s == 0:
                    continue   # sin deuda en este mes, no aporta
                otras[cuit][idx]["sits"].append(s)
                otras[cuit][idx]["montos"].append(meses[idx][1])
                otras[cuit][idx]["juds"].append(meses[idx][2])
                otras[cuit][idx]["tipos"].add(t)

    print(f"  Pasada 2 completa.")

    # ── CONSTRUCCIÓN DEL DATASET ────────────────────────────────────────────────
    print("\n[3] Construyendo features y targets...")

    rows = []
    n_obs = n_valida_a = n_valida_c = n_pos_a = n_pos_c = 0

    for cuit, s_meses in sant.items():
        sexo, rango = demograficos(cuit)
        o_meses = otras[cuit]   # lista de 24 acumuladores

        # censura por la izquierda: ya estaba en el mes más antiguo (idx=23)
        en_sant_mes23  = sit(s_meses, 23) > 0
        en_otras_mes23 = len(o_meses[23]["sits"]) > 0
        left_censored  = int(en_sant_mes23 or en_otras_mes23)

        for idx in range(24):
            # sin deuda en Santander este mes → no hay observación
            s_sant = sit(s_meses, idx)
            if s_sant == 0:
                continue

            # peor situación en otras entidades en t
            sits_o  = o_meses[idx]["sits"]
            peor_o  = max(sits_o) if sits_o else 0

            # peor situación en TODO el sistema (Santander + otras)
            peor_sistema = max(s_sant, peor_o)

            # condición de entrada por target
            valida_a = (peor_sistema == 1 and idx >= HORIZON_A)
            valida_c = (0 < peor_sistema <= 2 and idx >= HORIZON_C)

            if not valida_a and not valida_c:
                continue

            n_obs += 1

            # ── targets ───────────────────────────────────────────────────────
            # futuro = índices menores (idx-1, idx-2, ...)
            def peor_en(i):
                ss = sit(s_meses, i)
                so = o_meses[i]["sits"]
                return max([ss] + so) if (ss > 0 or so) else 0

            ta = tc = None
            if valida_a:
                ta = int(any(peor_en(idx - k) >= 2 for k in range(1, HORIZON_A + 1)))
                n_valida_a += 1
                n_pos_a    += ta
            if valida_c:
                tc = int(any(peor_en(idx - k) >= 3 for k in range(1, HORIZON_C + 1)))
                n_valida_c += 1
                n_pos_c    += tc

            # ── features Santander ────────────────────────────────────────────
            monto_t  = monto(s_meses, idx)
            monto_t3 = monto(s_meses, idx + 3)
            var_monto_3m = round((monto_t - monto_t3) / monto_t3 * 100, 2) \
                           if monto_t3 > 0 else None

            # ── features otras entidades ──────────────────────────────────────
            montos_o = o_meses[idx]["montos"]
            juds_o   = o_meses[idx]["juds"]
            tipos_o  = o_meses[idx]["tipos"]

            n_otras      = len(sits_o)
            n_sit2_o     = sum(1 for s in sits_o if s == 2)
            n_sit3plus_o = sum(1 for s in sits_o if s >= 3)
            monto_otras  = sum(montos_o)
            log_monto_o  = round(math.log1p(monto_otras), 4) if monto_otras > 0 else 0.0

            # ── features sistema total ────────────────────────────────────────
            # n entidades total = Santander (siempre 1) + otras activas
            n_total = 1 + n_otras

            # meses con deuda en sistema (desde idx hacia atrás)
            meses_deuda_sant = meses_con_deuda(s_meses, idx)
            meses_deuda_sis  = meses_deuda_sant + sum(
                1 for i in range(idx, 24)
                if len(o_meses[i]["sits"]) > 0
            )

            # log monto Santander
            log_monto_s = round(math.log1p(monto_t), 4) if monto_t > 0 else 0.0

            row = {
                "cuit":                      cuit,
                "periodo":                   PERIODOS[idx],
                # ── Santander ──
                "sit_sant_t":                s_sant,
                "sit_sant_t1":               sit(s_meses, idx + 1),
                "sit_sant_t2":               sit(s_meses, idx + 2),
                "sit_sant_t3":               sit(s_meses, idx + 3),
                "monto_sant_t":              round(monto_t, 1),
                "monto_sant_t3":             round(monto_t3, 1),
                "var_monto_3m_sant":         var_monto_3m,
                "log_monto_sant":            log_monto_s,
                "meses_consec_normal_sant":  meses_consec_normal(s_meses, idx),
                "meses_con_deuda_sant":      meses_deuda_sant,
                "judicial_sant_t":           jud(s_meses, idx),
                # ── otras entidades ──
                "n_entidades_otras_t":       n_otras,
                "peor_sit_otras_t":          peor_o,
                "n_ent_sit2_otras_t":        n_sit2_o,
                "n_ent_sit3plus_otras_t":    n_sit3plus_o,
                "monto_otras_t":             round(monto_otras, 1),
                "log_monto_otras":           log_monto_o,
                "tiene_judicial_otras_t":    int(any(j > 0 for j in juds_o)),
                "otras_tiene_banco_publico": int("banco_publico"    in tipos_o),
                "otras_tiene_banco_privado": int("banco_privado"    in tipos_o),
                "otras_tiene_banco_digital": int("banco_digital"    in tipos_o),
                "otras_tiene_financiera":    int("financiera"       in tipos_o),
                "otras_tiene_billetera":     int("billetera"        in tipos_o),
                "otras_tiene_prov_nofin":    int("prov_nofinanciero" in tipos_o),
                "otras_n_tipos_entidad":     len(tipos_o),
                "tiene_deuda_otras":         int(n_otras > 0),
                # ── sistema total ──
                "n_entidades_total_t":       n_total,
                "peor_sit_sistema_t":        peor_sistema,
                "meses_con_deuda_sistema":   meses_deuda_sis,
                # ── antigüedad ──
                "left_censored":             left_censored,
                # ── demografía ──
                "sexo_proxy":                sexo,
                "rango_edad_proxy":          rango,
                # ── targets ──
                "target_a":                  ta,
                "target_c":                  tc,
            }
            rows.append(row)

    print(f"  Observaciones totales  : {n_obs:,}")
    print(f"  Válidas target A       : {n_valida_a:,}  | positivos: {n_pos_a:,} ({n_pos_a/n_valida_a*100:.1f}%)" if n_valida_a else "  Target A: sin obs")
    print(f"  Válidas target C       : {n_valida_c:,}  | positivos: {n_pos_c:,} ({n_pos_c/n_valida_c*100:.1f}%)" if n_valida_c else "  Target C: sin obs")

    # ── guardar parquet ────────────────────────────────────────────────────────
    print(f"\n[4] Guardando {OUTPUT}...")

    # construir tabla columna por columna
    cols = list(rows[0].keys())
    data = {col: [r[col] for r in rows] for col in cols}

    # tipos explícitos para columnas críticas
    schema = pa.schema([
        ("cuit",                      pa.string()),
        ("periodo",                   pa.int32()),
        ("sit_sant_t",                pa.int8()),
        ("sit_sant_t1",               pa.int8()),
        ("sit_sant_t2",               pa.int8()),
        ("sit_sant_t3",               pa.int8()),
        ("monto_sant_t",              pa.float32()),
        ("monto_sant_t3",             pa.float32()),
        ("var_monto_3m_sant",         pa.float32()),
        ("log_monto_sant",            pa.float32()),
        ("meses_consec_normal_sant",  pa.int8()),
        ("meses_con_deuda_sant",      pa.int8()),
        ("judicial_sant_t",           pa.int8()),
        ("n_entidades_otras_t",       pa.int8()),
        ("peor_sit_otras_t",          pa.int8()),
        ("n_ent_sit2_otras_t",        pa.int8()),
        ("n_ent_sit3plus_otras_t",    pa.int8()),
        ("monto_otras_t",             pa.float32()),
        ("log_monto_otras",           pa.float32()),
        ("tiene_judicial_otras_t",    pa.int8()),
        ("otras_tiene_banco_publico", pa.int8()),
        ("otras_tiene_banco_privado", pa.int8()),
        ("otras_tiene_banco_digital", pa.int8()),
        ("otras_tiene_financiera",    pa.int8()),
        ("otras_tiene_billetera",     pa.int8()),
        ("otras_tiene_prov_nofin",    pa.int8()),
        ("otras_n_tipos_entidad",     pa.int8()),
        ("tiene_deuda_otras",         pa.int8()),
        ("n_entidades_total_t",       pa.int8()),
        ("peor_sit_sistema_t",        pa.int8()),
        ("meses_con_deuda_sistema",   pa.int8()),
        ("left_censored",             pa.int8()),
        ("sexo_proxy",                pa.string()),
        ("rango_edad_proxy",          pa.string()),
        ("target_a",                  pa.float32()),   # float para admitir None
        ("target_c",                  pa.float32()),
    ])

    tabla = pa.table(data, schema=schema)
    pq.write_table(tabla, OUTPUT, compression="snappy")

    tam = os.path.getsize(OUTPUT) / 1024**2
    print(f"  {OUTPUT}: {tam:.1f} MB | {len(rows):,} filas | {len(cols)} columnas")
    print(f"\nListo. Siguiente paso: python 03_split.py")


if __name__ == "__main__":
    main()

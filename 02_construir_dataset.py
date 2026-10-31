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
    - meses_consec_normal_sant_6m (racha sit=1 hacia atrás, tope 6)
    - judicial_sant_t
    - meses_con_deuda_sant_3m / _6m

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
    - meses_con_deuda_sistema_3m / _6m (meses con deuda en cualquier entidad)

  Demografía (derivada del CUIT):
    - sexo_proxy (M/F/I)
    - rango_edad_proxy

  Antigüedad:
    - deuda_sistema_t5 (ya tenía deuda en el sistema 5 meses antes, al inicio
      de la ventana de 6 meses)

Historia mínima: las features miran como mucho HISTORIA_MIN meses (t, t-1, ...,
t-5). Solo se generan filas para los períodos que tienen esa historia completa
dentro del archivo, así las features significan lo mismo en todas las filas y
en producción (donde siempre se predice con el mes más reciente).

Uso: python 02_construir_dataset.py
Tiempo estimado: ~20 min (dos pasadas sobre ~22GB), ~4GB de RAM
Requiere: pyarrow (pip install pyarrow)

Implementación pensada para correr en ~16GB de RAM:
  - Pasada 1: los 24 meses de cada CUIT de Santander se guardan en arrays
    compactos (sit / monto / judicial).
  - Pasada 2: por CUIT × mes se acumulan solo los agregados que usan las
    features (peor sit, n entidades, n sit2, n sit3+, judicial, tipos como
    bitmask, suma de montos). El acumulador se crea recién cuando el CUIT
    tiene deuda en otra entidad.
  - Las líneas se descartan por entidad / CUIT antes de parsear los 24 meses.
  - El parquet se escribe por bloques (ParquetWriter) a un .tmp que se
    renombra al terminar.
"""

import os
import math
from array import array
import pyarrow as pa
import pyarrow.parquet as pq

# ── configuración ───────────────────────────────────────────────────────────────
ARCHIVO_24DSF = "24DSF.txt"     # archivo agosto 2026 (24 meses: sep 2024 → ago 2026)
BASE_PERIODO  = 202608          # AAAAMM del mes índice 0 (más reciente)
ENTIDAD_FOCAL = "72"            # Santander Argentina
HORIZON_A     = 3               # meses a futuro para target_a
HORIZON_C     = 12              # meses a futuro para target_c
HISTORIA_MIN  = 6               # meses de historia (t incluido) que usan las features
VENTANA_CORTA = 3               # ventana corta de los conteos de meses con deuda
DNI_MAYOR_18  = 48_000_000      # DNI < este → mayor de 18 en 2026
DNI_EXTRANJERO = 90_000_000     # DNI >= este → extranjero, se excluye
OUTPUT        = "dataset_ews.parquet"
FILAS_POR_BLOQUE = 500_000      # filas acumuladas antes de escribir al parquet

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

# tipos como bits para guardarlos en un entero por CUIT × mes
BIT_TIPO = {
    "banco_publico":     1,
    "banco_privado":     2,
    "banco_digital":     4,
    "financiera":        8,
    "billetera":         16,
    "prov_nofinanciero": 32,
}


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


# ── parser de meses 24DSF ───────────────────────────────────────────────────────
def parsear_meses(line):
    """
    Estructura: 5 entidad | 2 tipo_id | 11 CUIT | 24 × (2 sit + 12 monto + 1 judicial)
    Retorna tres arrays (sits, montos, juds) de 24 posiciones.
    i=0 → mes más reciente, i=23 → más antiguo.
    """
    sits   = array("b", bytes(24))
    montos = array("d", bytes(24 * 8))
    juds   = array("b", bytes(24))
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
        sits[i], montos[i], juds[i] = s, m, j
    return sits, montos, juds


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
def get(arr, idx, default):
    return arr[idx] if 0 <= idx < 24 else default

def meses_consec_normal(sits, desde_idx, tope):
    """Meses consecutivos con sit=1 desde desde_idx hacia el pasado, hasta tope."""
    c, i = 0, desde_idx
    while c < tope and sits[i] == 1:
        c += 1
        i += 1
    return c

def meses_con_deuda(con_deuda, desde_idx, ventana):
    """Meses con deuda en los `ventana` meses que terminan en desde_idx (incluido)."""
    return sum(1 for i in range(desde_idx, desde_idx + ventana) if con_deuda(i))


# ── acumulador de otras entidades por CUIT ─────────────────────────────────────
# agg[k*24 + idx] para cada campo k:
F_N, F_PEOR, F_N2, F_N3P, F_JUD, F_TIPOS = range(6)

def nuevo_acumulador():
    # array de enteros (6 campos × 24 meses) + array de montos (24 meses)
    return array("i", bytes(6 * 24 * 4)), array("d", bytes(24 * 8))


# ── esquema de salida ──────────────────────────────────────────────────────────
SCHEMA = pa.schema([
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
    ("meses_consec_normal_sant_6m", pa.int8()),
    ("meses_con_deuda_sant_3m",   pa.int8()),
    ("meses_con_deuda_sant_6m",   pa.int8()),
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
    ("meses_con_deuda_sistema_3m", pa.int8()),
    ("meses_con_deuda_sistema_6m", pa.int8()),
    ("deuda_sistema_t5",          pa.int8()),
    ("sexo_proxy",                pa.string()),
    ("rango_edad_proxy",          pa.string()),
    ("target_a",                  pa.float32()),   # float para admitir None
    ("target_c",                  pa.float32()),
])
COLS = SCHEMA.names


# ── main ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Script 02 - Construcción dataset EWS v2")
    print("=" * 60)
    print(f"  Archivo    : {ARCHIVO_24DSF}")
    print(f"  Base período: {BASE_PERIODO}  ({PERIODOS[23]} → {PERIODOS[0]})")
    print(f"  Target A   : caída a sit>=2 en {HORIZON_A} meses")
    print(f"  Target C   : caída a sit>=3 en {HORIZON_C} meses")
    print(f"  Historia   : {HISTORIA_MIN} meses (filas desde {PERIODOS[24 - HISTORIA_MIN]})")
    print()

    if not os.path.exists(ARCHIVO_24DSF):
        print(f"ERROR: no se encuentra {ARCHIVO_24DSF}")
        return

    # ── PASADA 1: identificar CUITs elegibles de Santander ─────────────────────
    print("[1] Pasada 1 — CUITs de Santander con prefiltro...")
    sant = {}   # cuit → (sits, montos, juds)
    n = 0
    with open(ARCHIVO_24DSF, encoding="latin-1") as f:
        for line in f:
            n += 1
            if n % 5_000_000 == 0:
                print(f"  {n/1e6:.0f}M líneas | {len(sant):,} CUITs elegibles")
            line = line.rstrip("\r\n")
            if len(line) < 378:
                continue
            if line[0:5].strip() != ENTIDAD_FOCAL:
                continue
            cuit = line[7:18]
            if not es_elegible(cuit):
                continue
            meses = parsear_meses(line)
            if any(s > 0 for s in meses[0]):
                sant[cuit] = meses

    print(f"\n  Total líneas   : {n:,}")
    print(f"  CUITs elegibles: {len(sant):,}")

    # ── PASADA 2: acumular otras entidades para esos CUITs ─────────────────────
    # otras[cuit] = (agg, montos_sum); se crea solo si el CUIT tiene deuda en
    # alguna otra entidad. Ver F_* para el layout de agg.
    print(f"\n[2] Pasada 2 — acumulando otras entidades para {len(sant):,} CUITs...")
    otras = {}
    n = 0
    with open(ARCHIVO_24DSF, encoding="latin-1") as f:
        for line in f:
            n += 1
            if n % 5_000_000 == 0:
                print(f"  {n/1e6:.0f}M líneas | {len(otras):,} CUITs con otras entidades")
            line = line.rstrip("\r\n")
            if len(line) < 378:
                continue
            cuit = line[7:18]
            if cuit not in sant:
                continue
            entidad = line[0:5].strip()
            if entidad == ENTIDAD_FOCAL:
                continue
            bit = BIT_TIPO[tipo_entidad(entidad)]
            sits, montos, juds = parsear_meses(line)
            acc = None
            for idx in range(24):
                s = sits[idx]
                if s == 0:
                    continue   # sin deuda en este mes, no aporta
                if acc is None:
                    acc = otras.get(cuit)
                    if acc is None:
                        acc = otras[cuit] = nuevo_acumulador()
                agg, msum = acc
                if agg[F_N * 24 + idx] == 0 or s > agg[F_PEOR * 24 + idx]:
                    agg[F_PEOR * 24 + idx] = s
                agg[F_N * 24 + idx] += 1
                if s == 2:
                    agg[F_N2 * 24 + idx] += 1
                elif s >= 3:
                    agg[F_N3P * 24 + idx] += 1
                if juds[idx] > 0:
                    agg[F_JUD * 24 + idx] = 1
                agg[F_TIPOS * 24 + idx] |= bit
                msum[idx] += montos[idx]

    print(f"  Pasada 2 completa. {len(otras):,} CUITs con deuda en otras entidades.")

    # ── CONSTRUCCIÓN DEL DATASET ────────────────────────────────────────────────
    print("\n[3] Construyendo features y targets...")

    tmp_output = OUTPUT + ".tmp"
    writer = pq.ParquetWriter(tmp_output, SCHEMA, compression="snappy")
    bloque = {col: [] for col in COLS}
    filas_bloque = 0

    def volcar():
        nonlocal bloque, filas_bloque
        if filas_bloque:
            writer.write_table(pa.table(bloque, schema=SCHEMA))
            bloque = {col: [] for col in COLS}
            filas_bloque = 0

    vacio = nuevo_acumulador()   # para CUITs sin otras entidades
    n_obs = n_valida_a = n_valida_c = n_pos_a = n_pos_c = 0

    for cuit, (s_sits, s_montos, s_juds) in sant.items():
        sexo, rango = demograficos(cuit)
        agg, msum = otras.get(cuit, vacio)

        def n_o(i):    return agg[F_N * 24 + i]
        def peor_o(i): return agg[F_PEOR * 24 + i] if agg[F_N * 24 + i] else 0
        def deuda_sant(i):    return s_sits[i] > 0
        def deuda_sistema(i): return s_sits[i] > 0 or n_o(i) > 0

        # solo períodos con HISTORIA_MIN meses de historia dentro del archivo
        for idx in range(24 - HISTORIA_MIN + 1):
            # sin deuda en Santander este mes → no hay observación
            s_sant = s_sits[idx]
            if s_sant == 0:
                continue

            n_otras = n_o(idx)
            peor_ot = peor_o(idx)

            # peor situación en TODO el sistema (Santander + otras)
            peor_sistema = max(s_sant, peor_ot)

            # condición de entrada por target
            valida_a = (peor_sistema == 1 and idx >= HORIZON_A)
            valida_c = (0 < peor_sistema <= 2 and idx >= HORIZON_C)

            if not valida_a and not valida_c:
                continue

            n_obs += 1

            # ── targets ───────────────────────────────────────────────────────
            # futuro = índices menores (idx-1, idx-2, ...)
            def peor_en(i):
                ss = s_sits[i]
                if n_o(i):
                    return max(ss, peor_o(i))
                return ss if ss > 0 else 0

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
            monto_t  = s_montos[idx]
            monto_t3 = get(s_montos, idx + 3, 0.0)
            var_monto_3m = round((monto_t - monto_t3) / monto_t3 * 100, 2) \
                           if monto_t3 > 0 else None

            # ── features otras entidades ──────────────────────────────────────
            tipos_o      = agg[F_TIPOS * 24 + idx]
            monto_otras  = msum[idx]
            log_monto_o  = round(math.log1p(monto_otras), 4) if monto_otras > 0 else 0.0

            # ── features sistema total ────────────────────────────────────────
            n_total = 1 + n_otras

            log_monto_s = round(math.log1p(monto_t), 4) if monto_t > 0 else 0.0

            valores = (
                cuit,
                PERIODOS[idx],
                # ── Santander ──
                s_sant,
                get(s_sits, idx + 1, 0),
                get(s_sits, idx + 2, 0),
                get(s_sits, idx + 3, 0),
                round(monto_t, 1),
                round(monto_t3, 1),
                var_monto_3m,
                log_monto_s,
                meses_consec_normal(s_sits, idx, HISTORIA_MIN),
                meses_con_deuda(deuda_sant, idx, VENTANA_CORTA),
                meses_con_deuda(deuda_sant, idx, HISTORIA_MIN),
                s_juds[idx],
                # ── otras entidades ──
                n_otras,
                peor_ot,
                agg[F_N2 * 24 + idx],
                agg[F_N3P * 24 + idx],
                round(monto_otras, 1),
                log_monto_o,
                agg[F_JUD * 24 + idx],
                int(bool(tipos_o & BIT_TIPO["banco_publico"])),
                int(bool(tipos_o & BIT_TIPO["banco_privado"])),
                int(bool(tipos_o & BIT_TIPO["banco_digital"])),
                int(bool(tipos_o & BIT_TIPO["financiera"])),
                int(bool(tipos_o & BIT_TIPO["billetera"])),
                int(bool(tipos_o & BIT_TIPO["prov_nofinanciero"])),
                bin(tipos_o).count("1"),
                int(n_otras > 0),
                # ── sistema total ──
                n_total,
                peor_sistema,
                meses_con_deuda(deuda_sistema, idx, VENTANA_CORTA),
                meses_con_deuda(deuda_sistema, idx, HISTORIA_MIN),
                # ── antigüedad ──
                int(deuda_sistema(idx + HISTORIA_MIN - 1)),
                # ── demografía ──
                sexo,
                rango,
                # ── targets ──
                ta,
                tc,
            )
            assert len(valores) == len(COLS)
            for col, v in zip(COLS, valores):
                bloque[col].append(v)
            filas_bloque += 1
            if filas_bloque >= FILAS_POR_BLOQUE:
                volcar()

    volcar()
    writer.close()

    print(f"  Observaciones totales  : {n_obs:,}")
    print(f"  Válidas target A       : {n_valida_a:,}  | positivos: {n_pos_a:,} ({n_pos_a/n_valida_a*100:.1f}%)" if n_valida_a else "  Target A: sin obs")
    print(f"  Válidas target C       : {n_valida_c:,}  | positivos: {n_pos_c:,} ({n_pos_c/n_valida_c*100:.1f}%)" if n_valida_c else "  Target C: sin obs")

    # ── guardar parquet ────────────────────────────────────────────────────────
    print(f"\n[4] Guardando {OUTPUT}...")
    os.replace(tmp_output, OUTPUT)

    tam = os.path.getsize(OUTPUT) / 1024**2
    print(f"  {OUTPUT}: {tam:.1f} MB | {n_obs:,} filas | {len(COLS)} columnas")
    print(f"\nListo. Siguiente paso: python 03_split.py")


if __name__ == "__main__":
    main()

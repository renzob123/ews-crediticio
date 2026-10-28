"""
*** SUPERADO — NO USAR ***
Reemplazado por 02_construir_dataset.py (ver ese archivo), que hace el mismo trabajo
de extracción pero además calcula directamente features y targets en la misma pasada,
sin necesidad de guardar santander_raw.parquet / sistema_raw.parquet como paso intermedio.
Este archivo queda solo de referencia; podés borrarlo cuando confirmes que no lo necesitás.
(Nota agregada automáticamente — Claude, 2026-10-28)
*** FIN DE LA NOTA ***

Tesis EWS Crediticio - v2
Script 02: Extracción del 24DSF con prefiltro

Hace UNA sola pasada sobre el 24DSF y extrae únicamente los registros que
cumplen los tres criterios al mismo tiempo:
  1. Tienen deuda con entidad 72 (Santander Argentina) en algún mes
  2. Son personas físicas (prefijo CUIT 20, 23, 24 o 27 — excluye 30/33/34)
  3. Son mayores de edad estimados (rango DNI < 48.000.000 en año de referencia 2026)
     También excluye extranjeros (DNI >= 90.000.000)

El resultado se guarda en dos archivos parquet (con pyarrow):
  - santander_raw.parquet : una fila por CUIT, con los 24 meses de Santander
  - sistema_raw.parquet   : una fila por CUIT × entidad_otra, con sus 24 meses

Ventaja vs. v1: el filtro se aplica durante la lectura del archivo (pasada única),
lo que reduce el uso de RAM a solo los CUITs elegibles y evita guardar primero
todo el sistema para luego filtrar.

Configuración obligatoria: ajustar ARCHIVO_24DSF y BASE_PERIODO según el archivo
disponible. El nombre del archivo cambia cada mes (ej: 24DSF_202603.txt).

Uso: python 02_extraer_santander.py
Tiempo estimado: 20-40 min dependiendo del hardware (lectura secuencial de ~22GB)
"""

import os
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict

# ── configuración — AJUSTAR SEGÚN EL ARCHIVO DISPONIBLE ────────────────────────
ARCHIVO_24DSF = "24DSF.txt"    # ruta al archivo (puede ser relativa o absoluta)
BASE_PERIODO  = 202603         # AAAAMM del mes 0 (el más reciente en el archivo)
                               # ej: si el archivo es de marzo 2026 → 202603
                               # el archivo siempre tiene 24 meses hacia atrás

ENTIDAD_FOCAL = "72"           # Santander Argentina
DNI_MAYOR_18  = 48_000_000     # DNI < este valor → mayor de 18 en 2026
                               # (nacidos antes de 2008, aprox.)
DNI_EXTRANJERO = 90_000_000    # DNI >= este valor → extranjero, se excluye

OUTPUT_SANT   = "santander_raw.parquet"
OUTPUT_OTRAS  = "sistema_raw.parquet"
# ──────────────────────────────────────────────────────────────────────────────


def es_persona_fisica_mayor(cuit: str) -> bool:
    """
    Devuelve True si el CUIT corresponde a una persona física mayor de edad.
    Criterios:
      - Prefijo 20, 23, 24 o 27 (excluye jurídicas: 30, 33, 34)
      - DNI (dígitos 2-9) < DNI_MAYOR_18 (mayor de 18)
      - DNI < DNI_EXTRANJERO (no extranjero)
    """
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


def parsear(line: str):
    """
    Parsea una línea del 24DSF.
    Estructura: 5 entidad | 2 tipo_id | 11 CUIT | 24 × (2 sit + 12 monto + 1 judicial)
    Total: 5+2+11 + 24*15 = 378 caracteres mínimos.

    Retorna (entidad, cuit, meses) o None si la línea es inválida.
    meses[i] = (sit, monto_miles, judicial)
      i=0 → mes más reciente (BASE_PERIODO)
      i=23 → mes más antiguo (BASE_PERIODO - 23 meses)
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
            s = int(s_str) if s_str else 0
            m = float(m_str) if m_str else 0.0
            j = int(j_str) if j_str else 0
        except ValueError:
            s, m, j = 0, 0.0, 0
        meses.append((s, m, j))
    return entidad, cuit, meses


def meses_a_listas(meses):
    """Descompone la lista de tuplas en tres listas separadas (para parquet)."""
    sits   = [m[0] for m in meses]
    montos = [m[1] for m in meses]
    juds   = [m[2] for m in meses]
    return sits, montos, juds


def main():
    print("=" * 58)
    print("Script 02 - Extracción 24DSF con prefiltro")
    print("=" * 58)
    print(f"  Archivo    : {ARCHIVO_24DSF}")
    print(f"  Base período: {BASE_PERIODO}")
    print(f"  Entidad focal: {ENTIDAD_FOCAL} (Santander)")
    print(f"  Filtro     : personas físicas, DNI < {DNI_MAYOR_18:,}")
    print()

    if not os.path.exists(ARCHIVO_24DSF):
        print(f"ERROR: no se encuentra {ARCHIVO_24DSF}")
        print("Ajustá la variable ARCHIVO_24DSF en la configuración del script.")
        return

    # ── pasada única ────────────────────────────────────────────────────────────
    # En una sola pasada:
    #   - identificamos los CUITs que tienen deuda en Santander Y pasan el filtro
    #   - guardamos sus meses en Santander
    #   - guardamos los meses de las otras entidades para esos mismos CUITs
    #
    # Estructura en memoria:
    #   sant_data  : {cuit: meses_list}
    #   otras_data : {cuit: {entidad: meses_list}}  (solo CUITs ya en sant_data)
    #
    # Se hace en una sola pasada para no recorrer el archivo dos veces,
    # pero requiere mantener en RAM todas las "otras" entidades vistas
    # antes de que aparezca el registro de Santander para ese CUIT.
    # Para minimizar RAM, guardamos "otras" solo para los CUITs ya filtrados.
    # Los CUITs que aparecen en otras entidades antes que en Santander se
    # resuelven con un segundo pase liviano sobre un buffer de CUITs pendientes
    # (ver comentario más abajo).
    #
    # Alternativa sin buffer: dos pasadas cortas (la primera solo lee Santander
    # y filtra CUITs, la segunda lee otras). Se usa la variante de dos pasadas
    # si el dataset tiene demasiados CUITs pendientes (ver aviso).

    print("[1] Pasada 1 - identificando CUITs de Santander con prefiltro...")
    sant_data   = {}   # cuit → meses (solo Santander + filtro)
    n_lineas    = 0

    with open(ARCHIVO_24DSF, encoding="latin-1") as f:
        for line in f:
            n_lineas += 1
            if n_lineas % 5_000_000 == 0:
                print(f"  {n_lineas/1e6:.0f}M líneas | "
                      f"{len(sant_data):,} CUITs Santander elegibles")
            r = parsear(line.rstrip("\r\n"))
            if r is None:
                continue
            entidad, cuit, meses = r
            if entidad != ENTIDAD_FOCAL:
                continue
            # tiene deuda en Santander: aplicar filtro
            if not es_persona_fisica_mayor(cuit):
                continue
            # guardar solo si tiene al menos un mes con deuda (sit > 0)
            if any(m[0] > 0 for m in meses):
                sant_data[cuit] = meses

    print(f"\n  Líneas totales     : {n_lineas:,}")
    print(f"  CUITs elegibles    : {len(sant_data):,}")

    # ── pasada 2: recolectar otras entidades para los CUITs elegibles ────────────
    print(f"\n[2] Pasada 2 - recolectando sistema para {len(sant_data):,} CUITs...")
    otras_data  = defaultdict(dict)
    n_lineas    = 0

    with open(ARCHIVO_24DSF, encoding="latin-1") as f:
        for line in f:
            n_lineas += 1
            if n_lineas % 5_000_000 == 0:
                print(f"  {n_lineas/1e6:.0f}M líneas")
            r = parsear(line.rstrip("\r\n"))
            if r is None:
                continue
            entidad, cuit, meses = r
            # solo nos interesan otras entidades para CUITs ya elegibles
            if entidad == ENTIDAD_FOCAL or cuit not in sant_data:
                continue
            if any(m[0] > 0 for m in meses):
                otras_data[cuit][entidad] = meses

    print(f"  CUITs con deuda en otras entidades: {len(otras_data):,}")

    # ── guardar santander_raw.parquet ────────────────────────────────────────────
    print("\n[3] Guardando santander_raw.parquet...")

    rows_sant = []
    for cuit, meses in sant_data.items():
        sits, montos, juds = meses_a_listas(meses)
        rows_sant.append({
            "cuit":   cuit,
            "sits":   sits,    # lista de 24 ints
            "montos": montos,  # lista de 24 floats
            "juds":   juds,    # lista de 24 ints
        })

    tabla_sant = pa.Table.from_pydict({
        "cuit":   pa.array([r["cuit"]   for r in rows_sant], type=pa.string()),
        "sits":   pa.array([r["sits"]   for r in rows_sant], type=pa.list_(pa.int8())),
        "montos": pa.array([r["montos"] for r in rows_sant], type=pa.list_(pa.float32())),
        "juds":   pa.array([r["juds"]   for r in rows_sant], type=pa.list_(pa.int8())),
    })
    pq.write_table(tabla_sant, OUTPUT_SANT, compression="snappy")
    print(f"  {OUTPUT_SANT}: {len(rows_sant):,} filas")

    # ── guardar sistema_raw.parquet ──────────────────────────────────────────────
    print("\n[4] Guardando sistema_raw.parquet...")

    rows_otras = []
    for cuit, entidades in otras_data.items():
        for entidad, meses in entidades.items():
            sits, montos, juds = meses_a_listas(meses)
            rows_otras.append({
                "cuit":    cuit,
                "entidad": entidad,
                "sits":    sits,
                "montos":  montos,
                "juds":    juds,
            })

    tabla_otras = pa.Table.from_pydict({
        "cuit":    pa.array([r["cuit"]    for r in rows_otras], type=pa.string()),
        "entidad": pa.array([r["entidad"] for r in rows_otras], type=pa.string()),
        "sits":    pa.array([r["sits"]    for r in rows_otras], type=pa.list_(pa.int8())),
        "montos":  pa.array([r["montos"]  for r in rows_otras], type=pa.list_(pa.float32())),
        "juds":    pa.array([r["juds"]    for r in rows_otras], type=pa.list_(pa.int8())),
    })
    pq.write_table(tabla_otras, OUTPUT_OTRAS, compression="snappy")
    print(f"  {OUTPUT_OTRAS}: {len(rows_otras):,} filas")

    # ── resumen final ────────────────────────────────────────────────────────────
    tamaño_sant  = os.path.getsize(OUTPUT_SANT)  / 1024**2
    tamaño_otras = os.path.getsize(OUTPUT_OTRAS) / 1024**2
    print(f"\n{'='*58}")
    print(f"Listo.")
    print(f"  {OUTPUT_SANT}  : {tamaño_sant:.1f} MB")
    print(f"  {OUTPUT_OTRAS} : {tamaño_otras:.1f} MB")
    print(f"  CUITs totales  : {len(sant_data):,}")
    print(f"  Pares CUIT×ent : {len(rows_otras):,}")
    print(f"\nSiguiente paso: python 03_construir_dataset.py")


if __name__ == "__main__":
    main()

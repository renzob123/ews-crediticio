"""
Tesis EWS Crediticio - v2 (Tesis_v2)
Script 03: Features derivadas + macro + split train/test

Toma dataset_ews.parquet (salida de 02_construir_dataset.py) y macro_bcra.csv
(salida de 01_descargar_macro_bcra.py) y:

  1. Montos reales (duda 2 de docs/decisiones_metodologicas.md): deflacta con el
     CER los montos nominales de 02 y los expresa en pesos del último período del
     dataset. Cada monto usa el CER de SU mes (monto_*_t3 usa el CER de t-3).
       monto_sant_t_real, monto_sant_t3_real, var_monto_3m_sant_real,
       log_monto_sant_real, monto_otras_t_real, monto_otras_t3_real,
       log_monto_otras_real
  2. Features libres de inflación:
       share_sant_t, share_sant_t3 = monto_sant / (monto_sant + monto_otras)
       dec_monto_sant_t, dec_monto_sant_t3, dec_monto_otras_t, dec_monto_otras_t3:
         decil del monto dentro de su período. 0 = sin deuda (monto 0), 1-10 entre
         los montos > 0. Los cortes de cada período se calculan en una primera
         pasada con las filas en sit 1 en todo el sistema (la población de los
         targets A y B), para que el decil signifique lo mismo en todos los
         períodos. No usan el target ni datos de otros meses, así que se calculan
         con todas las filas (train y test), igual que haría el banco al puntuar.
  3. Macro (duda 5): las 10 variables del modelo, cada una con el rezago que le
     corresponde según qué estaba publicado al momento del scoring (fines de
     t+1). Ver REZAGO_MACRO. Los meses que el BCRA no publicó quedan vacíos.
  4. Split train/test a NIVEL CUIT (no fila), estratificado por "alguna vez
     target_c == 1", para que las filas de un mismo CUIT no queden partidas.
  5. Validación fuera de tiempo: una columna oot_* por target (incluidos los de
     sensibilidad _t1) con "train", "hueco" o "test" según el período:
       - test : los últimos OOT_MESES_TEST períodos con el target válido
       - train: períodos cuya ventana del target termina a más tardar en el
                primer mes de test (periodo + fin de ventana <= primer mes de
                test); ese mes solo aporta features al test, no su respuesta
       - hueco: el resto; su respuesta depende de meses del test
     La ventana de cada target (inicio y horizonte) se lee de los metadatos que
     escribe 02, así no hay que mantener los valores en dos lugares.

La evaluación más conservadora combina los dos splits: entrenar con
dataset_train.parquet & oot == "train", evaluar con dataset_test.parquet &
oot == "test" (otras personas y meses posteriores).

Memoria: el dataset tiene ~35M filas, así que no se carga entero. Se leen solo
las columnas necesarias para el split y los deciles, y después el parquet se
recorre por row group: a cada bloque se le agregan las columnas y se escribe en
train o test.

Uso: python 03_split.py
Requiere: pandas, pyarrow
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── paths (independientes del directorio desde donde se ejecute) ──────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent

DATASET_EWS   = _SCRIPT_DIR / "dataset_ews.parquet"      # salida de 02_construir_dataset.py
MACRO_CSV     = _SCRIPT_DIR / "macro_bcra.csv"            # salida de 01_descargar_macro_bcra.py
OUT_TRAIN     = _SCRIPT_DIR / "dataset_train.parquet"
OUT_TEST      = _SCRIPT_DIR / "dataset_test.parquet"

TEST_SIZE     = 0.20     # proporción de CUITs que van a test
RANDOM_STATE  = 42        # fijo para reproducibilidad

OOT_MESES_TEST = 3        # últimos períodos válidos de cada target que van a test
OOT_CATEGORIAS = ["train", "hueco", "test"]

# Variables macro del modelo → meses de rezago (0 = mes t, 1 = mes t-1).
# Regla: t si el dato de t ya estaba publicado a fines de t+1 (momento del
# scoring), t-1 si no. Según las descargas del 27/10/2026 y del 18/11/2026.
REZAGO_MACRO = {
    "mora_3plus_privado_pct":    0,
    "sit2_privado_pct":          0,
    "var_mora3plus_3m":          0,
    "var_sit2_3m":               0,
    "tasa_consumo_corto_pct":    0,
    "badlar_privados_tna_pct":   0,
    "spread_consumo_badlar_pct": 0,
    "var_tc_3m_pct":             0,
    "pct_personales":            0,
    "var_credito_real_3m_pct":   1,
}

# montos nominales de 02 → (columna real, meses hacia atrás del CER a usar)
MONTOS = {
    "monto_sant_t":   ("monto_sant_t_real",   0),
    "monto_sant_t3":  ("monto_sant_t3_real",  3),
    "monto_otras_t":  ("monto_otras_t_real",  0),
    "monto_otras_t3": ("monto_otras_t3_real", 3),
}
DECILES = {col: "dec_" + col for col in MONTOS}
CORTES_Q = np.linspace(0.1, 0.9, 9)


def sumar_meses(periodo: int, k: int) -> int:
    total = (periodo // 100) * 12 + periodo % 100 - 1 + k
    return (total // 12) * 100 + total % 12 + 1


# ── parámetros de las ventanas (metadatos de 02) ─────────────────────────────────
def leer_ventanas(pf: pq.ParquetFile) -> dict:
    """Devuelve {target: (columna oot, último mes de la ventana)}."""
    meta = {k.decode(): v.decode() for k, v in (pf.schema_arrow.metadata or {}).items()}
    if "ews_horizontes" not in meta:
        print("ERROR: dataset_ews.parquet no tiene los metadatos de ventanas. "
              "Volvé a correr 02_construir_dataset.py.")
        sys.exit(1)
    horizontes = dict(kv.split("=") for kv in meta["ews_horizontes"].split(","))
    inicios = {"": int(meta["ews_inicio_ventana"]), "_t1": int(meta["ews_inicio_ventana_sens"])}
    ventanas = {}
    for suf, ini in inicios.items():
        for target, h in horizontes.items():
            col = target + suf
            ventanas[col] = ("oot_" + col.removeprefix("target_"), ini + int(h) - 1)
    return ventanas


# ── macro ──────────────────────────────────────────────────────────────────────
def cargar_macro(periodos: np.ndarray):
    """
    Devuelve (macro_por_periodo, cer) o (None, None) si no hay macro_bcra.csv.
    macro_por_periodo: una fila por período del dataset con las variables del
    modelo ya rezagadas. cer: Series periodo → CER.
    """
    if not MACRO_CSV.exists():
        print(f"ERROR: no se encontró {MACRO_CSV.name}. Corré primero 01_descargar_macro_bcra.py")
        sys.exit(1)

    macro = pd.read_csv(MACRO_CSV, sep=";").set_index("periodo_aaaamm")
    faltan = [c for c in [*REZAGO_MACRO, "cer"] if c not in macro.columns]
    if faltan:
        print(f"ERROR: a {MACRO_CSV.name} le faltan columnas {faltan}. "
              f"Volvé a correr 01_descargar_macro_bcra.py")
        sys.exit(1)

    out = pd.DataFrame({"periodo": np.sort(periodos)})
    for col, rezago in REZAGO_MACRO.items():
        fuente = out["periodo"].map(lambda p: sumar_meses(int(p), -rezago))
        out[col] = fuente.map(macro[col]).astype("float32")

    print(f"  Macro: +{len(REZAGO_MACRO)} columnas "
          f"(rezago t-1 en: {[c for c, r in REZAGO_MACRO.items() if r] or 'ninguna'}).")
    vacios = out.set_index("periodo").isna()
    for col in REZAGO_MACRO:
        ps = vacios.index[vacios[col]].tolist()
        if ps:
            print(f"    {col:<28} vacía en {ps}")
    return out, macro["cer"].dropna()


# ── montos reales, shares y deciles ─────────────────────────────────────────────
def factores_cer(periodos: np.ndarray, cer: pd.Series) -> dict:
    """
    Para cada columna de monto, {periodo: factor} tal que monto * factor = monto
    en pesos del último período del dataset.
    """
    base = int(periodos.max())
    if base not in cer.index:
        print(f"ERROR: falta el CER de {base} en {MACRO_CSV.name}")
        sys.exit(1)
    factores = {}
    for col, (_, atras) in MONTOS.items():
        f = {}
        for p in periodos:
            p_cer = sumar_meses(int(p), -atras)
            if p_cer not in cer.index:
                print(f"ERROR: falta el CER de {p_cer} en {MACRO_CSV.name}")
                sys.exit(1)
            f[int(p)] = cer[base] / cer[p_cer]
        factores[col] = f
    print(f"  Montos reales en pesos de {base} (CER {cer[base]:.2f}).")
    return factores


def cortes_deciles() -> dict:
    """Pasada 1: {columna: {periodo: 9 cortes}} con las filas en sit 1 y monto > 0."""
    tabla = pd.read_parquet(DATASET_EWS, columns=["periodo", "peor_sit_sistema_t", *MONTOS])
    tabla = tabla[tabla["peor_sit_sistema_t"] == 1]
    cortes = {}
    for col in MONTOS:
        pos = tabla.loc[tabla[col] > 0, ["periodo", col]]
        cortes[col] = {int(p): np.quantile(g[col].to_numpy(), CORTES_Q)
                       for p, g in pos.groupby("periodo")}
    print(f"  Cortes de deciles calculados sobre {len(tabla):,} filas en sit 1 "
          f"({tabla['periodo'].nunique()} períodos).")
    return cortes


def asignar_decil(montos: pd.Series, periodos: pd.Series, cortes: dict) -> np.ndarray:
    dec = np.zeros(len(montos), dtype="int8")
    m, per = montos.to_numpy(), periodos.to_numpy()
    for p in np.unique(per):
        idx = np.flatnonzero((per == p) & (m > 0))
        if len(idx):
            c = cortes.get(int(p))
            if c is None:
                raise ValueError(f"sin cortes de deciles para el período {p}")
            dec[idx] = np.minimum(np.searchsorted(c, m[idx], side="right") + 1, 10)
    return dec


def agregar_derivadas(df: pd.DataFrame, factores: dict, cortes: dict) -> pd.DataFrame:
    for col, (col_real, _) in MONTOS.items():
        df[col_real] = (df[col] * df["periodo"].map(factores[col])).astype("float32")
    t, t3 = df["monto_sant_t_real"], df["monto_sant_t3_real"]
    df["var_monto_3m_sant_real"] = ((t - t3) / t3 * 100).where(t3 > 0).astype("float32")
    df["log_monto_sant_real"]  = np.log1p(t.clip(lower=0)).astype("float32")
    df["log_monto_otras_real"] = np.log1p(df["monto_otras_t_real"].clip(lower=0)).astype("float32")

    for suf in ("t", "t3"):
        s, o = df[f"monto_sant_{suf}"], df[f"monto_otras_{suf}"]
        total = s + o
        df[f"share_sant_{suf}"] = (s / total).where(total > 0).astype("float32")

    for col, col_dec in DECILES.items():
        df[col_dec] = asignar_decil(df[col], df["periodo"], cortes[col])
    return df


# ── validación fuera de tiempo ─────────────────────────────────────────────────
def etiquetas_oot(ids: pd.DataFrame, ventanas: dict) -> dict:
    """Para cada target, devuelve {periodo: "train" | "hueco" | "test"}."""
    etiquetas = {}
    for target, (col, fin) in ventanas.items():
        validos = sorted(int(p) for p in ids.loc[ids[target].notna(), "periodo"].unique())
        primer_test = validos[-OOT_MESES_TEST]
        limite_train = sumar_meses(primer_test, -fin)   # su ventana termina a más tardar en primer_test
        etiq = {p: "test" if p >= primer_test else "train" if p <= limite_train else "hueco"
                for p in validos}
        etiquetas[target] = etiq

        por_etiqueta = {e: [p for p in validos if etiq[p] == e] for e in OOT_CATEGORIAS}
        filas = ids["periodo"].map(etiq).where(ids[target].notna()).value_counts()
        print(f"  {col} (ventana hasta t+{fin}):")
        for e in OOT_CATEGORIAS:
            ps = por_etiqueta[e]
            rango = f"{ps[0]}-{ps[-1]} ({len(ps)} meses)" if ps else "ninguno"
            print(f"    {e:<6}: {rango:<26} {filas.get(e, 0):>12,} filas")
        if not por_etiqueta["train"]:
            print(f"    AVISO: no hay períodos de train fuera de tiempo para {target} "
                  f"(hacen falta más meses de historia).")
    return etiquetas


def agregar_oot(df: pd.DataFrame, etiquetas: dict, ventanas: dict) -> pd.DataFrame:
    for target, (col, _) in ventanas.items():
        df[col] = pd.Categorical(df["periodo"].map(etiquetas[target]).where(df[target].notna()),
                                 categories=OOT_CATEGORIAS)
    return df


# ── split por CUIT ─────────────────────────────────────────────────────────────
def elegir_cuits_test(ids: pd.DataFrame, test_size: float, seed: int) -> set:
    """
    Split a nivel CUIT (no fila), estratificado por 'alguna vez target_c == 1'.
    Devuelve el set de CUITs de test.
    """
    rng = np.random.default_rng(seed)

    cuit_flag = (
        ids.groupby("cuit")["target_c"]
        .max()
        .fillna(0)
        .gt(0)
        .rename("ever_target_c")
        .reset_index()
    )

    cuits_pos = cuit_flag.loc[cuit_flag.ever_target_c, "cuit"].to_numpy()
    cuits_neg = cuit_flag.loc[~cuit_flag.ever_target_c, "cuit"].to_numpy()

    def elegir_test(cuits):
        n_test = int(round(len(cuits) * test_size))
        idx = rng.permutation(len(cuits))
        return set(cuits[idx[:n_test]])

    test_cuits = elegir_test(cuits_pos) | elegir_test(cuits_neg)

    print(f"  CUITs totales      : {len(cuit_flag):,} "
          f"({len(cuits_pos):,} con target_c=1 alguna vez, {len(cuits_neg):,} sin)")
    print(f"  CUITs en test      : {len(test_cuits):,} ({len(test_cuits)/len(cuit_flag)*100:.1f}%)")

    return test_cuits


def main():
    print("=" * 58)
    print("Script 03 - Features derivadas + macro + split (Tesis_v2)")
    print("=" * 58)

    if not DATASET_EWS.exists():
        print(f"ERROR: no se encuentra {DATASET_EWS.name}")
        print("Corré primero 02_construir_dataset.py")
        sys.exit(1)

    pf = pq.ParquetFile(DATASET_EWS)
    ventanas = leer_ventanas(pf)

    print(f"\n[1] Cargando cuit / periodo / targets de {DATASET_EWS.name}...")
    ids = pd.read_parquet(DATASET_EWS, columns=["cuit", "periodo", *ventanas])
    print(f"  {len(ids):,} filas, {ids['cuit'].nunique():,} CUITs")
    periodos = ids["periodo"].unique()

    print(f"\n[2] Macro y CER...")
    macro, cer = cargar_macro(periodos)
    factores = factores_cer(periodos, cer)

    print(f"\n[3] Split train/test a nivel CUIT (test_size={TEST_SIZE}, seed={RANDOM_STATE})...")
    test_cuits = elegir_cuits_test(ids, TEST_SIZE, RANDOM_STATE)

    print(f"\n[4] Validación fuera de tiempo (últimos {OOT_MESES_TEST} meses válidos a test)...")
    etiquetas = etiquetas_oot(ids, ventanas)
    del ids

    print(f"\n[5] Cortes de deciles por período...")
    cortes = cortes_deciles()

    print(f"\n[6] Agregando columnas y guardando por bloques...")
    tmp = {OUT_TRAIN: OUT_TRAIN.with_name(OUT_TRAIN.name + ".tmp"),
           OUT_TEST:  OUT_TEST.with_name(OUT_TEST.name + ".tmp")}
    writers = {}
    filas = {OUT_TRAIN: 0, OUT_TEST: 0}
    sin_macro = 0

    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i).to_pandas()
        antes = len(df)
        df = agregar_derivadas(df, factores, cortes)
        df = df.merge(macro, on="periodo", how="left")
        assert len(df) == antes, "el merge con macro no debería cambiar la cantidad de filas"
        df = agregar_oot(df, etiquetas, ventanas)
        sin_macro += df[list(REZAGO_MACRO)].isna().all(axis=1).sum()

        es_test = df["cuit"].isin(test_cuits)
        for out, parte in ((OUT_TRAIN, df.loc[~es_test]), (OUT_TEST, df.loc[es_test])):
            if parte.empty:
                continue
            tabla = pa.Table.from_pandas(parte, preserve_index=False)
            if out not in writers:
                writers[out] = pq.ParquetWriter(tmp[out], tabla.schema, compression="snappy")
            else:
                tabla = tabla.cast(writers[out].schema)
            writers[out].write_table(tabla)
            filas[out] += len(parte)

    for w in writers.values():
        w.close()
    for out in writers:
        os.replace(tmp[out], out)

    if sin_macro:
        print(f"  AVISO: {sin_macro:,} filas quedaron sin ningún dato macro.")
    n_cols = pq.ParquetFile(OUT_TRAIN).schema_arrow.__len__()
    print(f"  Filas train / test : {filas[OUT_TRAIN]:,} / {filas[OUT_TEST]:,} ({n_cols} columnas)")
    print(f"  {OUT_TRAIN.name}: {OUT_TRAIN.stat().st_size / 1024**2:.1f} MB")
    print(f"  {OUT_TEST.name}: {OUT_TEST.stat().st_size / 1024**2:.1f} MB")

    print("\nListo. Siguiente paso: modelado (04_modelado.py) sobre dataset_train.parquet /")
    print("dataset_test.parquet.")


if __name__ == "__main__":
    main()

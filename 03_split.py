"""
Tesis EWS Crediticio - v2 (Tesis_v2)
Script 03: Merge de macro + split train/test

Toma dataset_ews.parquet (salida de 02_construir_dataset.py) y:
  1. Le suma las columnas macro de macro_bcra.csv (salida de 01_descargar_macro_bcra.py),
     haciendo join por "periodo" (AAAAMM) == "periodo_aaaamm". El join es left: las filas
     sin macro disponible (por ejemplo períodos fuera de rango) quedan con NaN en esas
     columnas, no se descartan.
  2. Separa en train/test a NIVEL CUIT (no a nivel fila), para que las observaciones
     mensuales repetidas de un mismo CUIT no queden partidas entre train y test
     (fuga de información). Se estratifica por "alguna vez target_c == 1" por CUIT,
     igual que en pipeline_v2/05_split_train_test.py.
  3. Agrega columnas de validación fuera de tiempo (oot_a, oot_c) que marcan cada fila
     según su período como "train", "hueco" o "test" para ese target:
       - test : los últimos OOT_MESES_TEST períodos con el target válido
       - train: períodos cuya ventana del target termina antes del primer mes de test
                (periodo + horizonte <= primer mes de test)
       - hueco: el resto; no se usan, porque su respuesta depende de meses del test
     Son NaN en las filas donde ese target no es válido.

     La evaluación más conservadora combina los dos splits: entrenar con
     dataset_train.parquet & oot == "train", evaluar con dataset_test.parquet &
     oot == "test" (otras personas y meses posteriores).

Nota: las columnas macro quedan en el dataset pero no implican que haya que usarlas en
el modelo base — en pipeline_v2 se entrenó dos veces (con y sin macro) justamente porque
el análisis de PSI mostró inestabilidad temporal fuerte en esas variables. Dejarlas acá
simplemente te da la opción de repetir esa comparación con la data de agosto 2026.

Memoria: el dataset tiene ~32M filas y el macro ~48 columnas, así que mergear todo en
memoria no entra en 16GB. Por eso el split se decide leyendo solo cuit + target_c, y
después el parquet se recorre por row group: a cada bloque se le pega el macro y se
escribe en train o test. El resultado es el mismo que hacer todo en memoria.

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

# validación fuera de tiempo: target → (columna, horizonte en meses)
OOT = {
    "target_a": ("oot_a", 3),
    "target_c": ("oot_c", 12),
}
OOT_MESES_TEST = 3        # últimos períodos válidos de cada target que van a test
OOT_CATEGORIAS = ["train", "hueco", "test"]


def cargar_macro(periodos: np.ndarray):
    """
    Devuelve (macro, dtypes) o (None, None) si no hay macro_bcra.csv.
    dtypes son los tipos que tendría el merge sobre el dataset completo (por ejemplo,
    una columna entera pasa a float si algún período queda sin macro); se aplican a
    cada bloque para que todos se escriban con el mismo esquema.
    """
    if not MACRO_CSV.exists():
        print(f"  AVISO: no se encontró {MACRO_CSV.name} — se continúa sin columnas macro.")
        print(f"         (correr primero 01_descargar_macro_bcra.py si las querés incluir)")
        return None, None

    macro = pd.read_csv(MACRO_CSV, sep=";")
    macro = macro.rename(columns={"periodo_aaaamm": "periodo"})
    macro = macro.drop(columns=["fecha_bcra"], errors="ignore")

    por_periodo = pd.DataFrame({"periodo": periodos}).merge(macro, on="periodo", how="left")
    assert len(por_periodo) == len(periodos), \
        "el merge con macro no debería cambiar la cantidad de filas"

    print(f"  Macro: +{len(macro.columns) - 1} columnas.")
    return macro, por_periodo.dtypes


def merge_macro(df: pd.DataFrame, macro, dtypes) -> pd.DataFrame:
    if macro is None:
        return df
    antes = len(df)
    df = df.merge(macro, on="periodo", how="left")
    assert len(df) == antes, "el merge con macro no debería cambiar la cantidad de filas"
    return df.astype(dtypes.to_dict())


def sumar_meses(periodo: int, k: int) -> int:
    total = (periodo // 100) * 12 + periodo % 100 - 1 + k
    return (total // 12) * 100 + total % 12 + 1


def etiquetas_oot(ids: pd.DataFrame) -> dict:
    """Para cada target, devuelve {periodo: "train" | "hueco" | "test"}."""
    etiquetas = {}
    for target, (col, horizonte) in OOT.items():
        validos = sorted(int(p) for p in ids.loc[ids[target].notna(), "periodo"].unique())
        primer_test = validos[-OOT_MESES_TEST]
        limite_train = sumar_meses(primer_test, -horizonte)
        etiq = {p: "test" if p >= primer_test else "train" if p <= limite_train else "hueco"
                for p in validos}
        etiquetas[target] = etiq

        por_etiqueta = {e: [p for p in validos if etiq[p] == e] for e in OOT_CATEGORIAS}
        filas = ids["periodo"].map(etiq).where(ids[target].notna()).value_counts()
        print(f"  {col} (horizonte {horizonte} meses):")
        for e in OOT_CATEGORIAS:
            ps = por_etiqueta[e]
            rango = f"{ps[0]}-{ps[-1]} ({len(ps)} meses)" if ps else "ninguno"
            print(f"    {e:<6}: {rango:<26} {filas.get(e, 0):>12,} filas")
        if not por_etiqueta["train"]:
            print(f"    AVISO: con este archivo no hay períodos de train fuera de tiempo "
                  f"para {target} (hacen falta más meses de historia).")
    return etiquetas


def agregar_oot(df: pd.DataFrame, etiquetas: dict) -> pd.DataFrame:
    for target, (col, _) in OOT.items():
        df[col] = pd.Categorical(df["periodo"].map(etiquetas[target]).where(df[target].notna()),
                                 categories=OOT_CATEGORIAS)
    return df


def elegir_cuits_test(ids: pd.DataFrame, test_size: float, seed: int) -> set:
    """
    Split a nivel CUIT (no fila), estratificado por 'alguna vez target_c == 1'.
    Recibe solo las columnas cuit y target_c; devuelve el set de CUITs de test.
    """
    rng = np.random.default_rng(seed)

    # flag por CUIT: ¿tuvo target_c=1 en algún período?
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
    print("Script 03 - Merge macro + split train/test (Tesis_v2)")
    print("=" * 58)

    if not DATASET_EWS.exists():
        print(f"ERROR: no se encuentra {DATASET_EWS.name}")
        print("Corré primero 02_construir_dataset.py")
        sys.exit(1)

    print(f"\n[1] Cargando cuit / periodo / targets de {DATASET_EWS.name}...")
    ids = pd.read_parquet(DATASET_EWS, columns=["cuit", "periodo", "target_a", "target_c"])
    print(f"  {len(ids):,} filas, {ids['cuit'].nunique():,} CUITs")
    periodos = ids["periodo"].unique()

    print(f"\n[2] Macro...")
    macro, dtypes = cargar_macro(periodos)

    print(f"\n[3] Split train/test a nivel CUIT (test_size={TEST_SIZE}, seed={RANDOM_STATE})...")
    test_cuits = elegir_cuits_test(ids, TEST_SIZE, RANDOM_STATE)

    print(f"\n[4] Validación fuera de tiempo (últimos {OOT_MESES_TEST} meses válidos a test)...")
    etiquetas = etiquetas_oot(ids)
    del ids

    print(f"\n[5] Mergeando macro y guardando por bloques...")
    pf = pq.ParquetFile(DATASET_EWS)
    tmp = {OUT_TRAIN: OUT_TRAIN.with_name(OUT_TRAIN.name + ".tmp"),
           OUT_TEST:  OUT_TEST.with_name(OUT_TEST.name + ".tmp")}
    writers = {}
    filas = {OUT_TRAIN: 0, OUT_TEST: 0}
    sin_macro = 0

    for i in range(pf.num_row_groups):
        df = merge_macro(pf.read_row_group(i).to_pandas(), macro, dtypes)
        df = agregar_oot(df, etiquetas)
        if macro is not None:
            sin_macro += df[macro.columns.drop("periodo")].isna().all(axis=1).sum()

        es_test = df["cuit"].isin(test_cuits)
        for out, parte in ((OUT_TRAIN, df.loc[~es_test]), (OUT_TEST, df.loc[es_test])):
            if parte.empty:
                continue
            tabla = pa.Table.from_pandas(parte, preserve_index=False)
            if out not in writers:
                writers[out] = pq.ParquetWriter(tmp[out], tabla.schema, compression="snappy")
            writers[out].write_table(tabla)
            filas[out] += len(parte)

    for w in writers.values():
        w.close()
    for out in writers:
        os.replace(tmp[out], out)

    if sin_macro:
        print(f"  AVISO: {sin_macro:,} filas quedaron sin datos macro "
              f"(período fuera del rango de macro_bcra.csv).")
    print(f"  Filas train / test : {filas[OUT_TRAIN]:,} / {filas[OUT_TEST]:,}")
    print(f"  {OUT_TRAIN.name}: {OUT_TRAIN.stat().st_size / 1024**2:.1f} MB")
    print(f"  {OUT_TEST.name}: {OUT_TEST.stat().st_size / 1024**2:.1f} MB")

    print("\nListo. Siguiente paso: correr el modelado (adaptar 09_modelado_ews.py de la v1,")
    print("o escribir un 04_modelado.py propio de Tesis_v2) sobre dataset_train.parquet /")
    print("dataset_test.parquet.")


if __name__ == "__main__":
    main()

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

Nota: las columnas macro quedan en el dataset pero no implican que haya que usarlas en
el modelo base — en pipeline_v2 se entrenó dos veces (con y sin macro) justamente porque
el análisis de PSI mostró inestabilidad temporal fuerte en esas variables. Dejarlas acá
simplemente te da la opción de repetir esa comparación con la data de agosto 2026.

Uso: python 03_split.py
Requiere: pandas, pyarrow
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths (independientes del directorio desde donde se ejecute) ──────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent

DATASET_EWS   = _SCRIPT_DIR / "dataset_ews.parquet"      # salida de 02_construir_dataset.py
MACRO_CSV     = _SCRIPT_DIR / "macro_bcra.csv"            # salida de 01_descargar_macro_bcra.py
OUT_TRAIN     = _SCRIPT_DIR / "dataset_train.parquet"
OUT_TEST      = _SCRIPT_DIR / "dataset_test.parquet"

TEST_SIZE     = 0.20     # proporción de CUITs que van a test
RANDOM_STATE  = 42        # fijo para reproducibilidad


def merge_macro(df: pd.DataFrame) -> pd.DataFrame:
    if not MACRO_CSV.exists():
        print(f"  AVISO: no se encontró {MACRO_CSV.name} — se continúa sin columnas macro.")
        print(f"         (correr primero 01_descargar_macro_bcra.py si las querés incluir)")
        return df

    macro = pd.read_csv(MACRO_CSV, sep=";")
    macro = macro.rename(columns={"periodo_aaaamm": "periodo"})
    macro = macro.drop(columns=["fecha_bcra"], errors="ignore")

    antes = len(df)
    df = df.merge(macro, on="periodo", how="left")
    assert len(df) == antes, "el merge con macro no debería cambiar la cantidad de filas"

    sin_macro = df[macro.columns.drop("periodo")].isna().all(axis=1).sum()
    if sin_macro:
        print(f"  AVISO: {sin_macro:,} filas quedaron sin datos macro "
              f"(período fuera del rango de macro_bcra.csv).")
    print(f"  Macro mergeado: +{len(macro.columns) - 1} columnas.")
    return df


def split_por_cuit(df: pd.DataFrame, test_size: float, seed: int):
    """
    Split a nivel CUIT (no fila), estratificado por 'alguna vez target_c == 1'.
    Devuelve (df_train, df_test).
    """
    rng = np.random.default_rng(seed)

    # flag por CUIT: ¿tuvo target_c=1 en algún período?
    cuit_flag = (
        df.groupby("cuit")["target_c"]
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

    es_test = df["cuit"].isin(test_cuits)
    df_train = df.loc[~es_test].reset_index(drop=True)
    df_test  = df.loc[es_test].reset_index(drop=True)

    print(f"  CUITs totales      : {len(cuit_flag):,} "
          f"({len(cuits_pos):,} con target_c=1 alguna vez, {len(cuits_neg):,} sin)")
    print(f"  CUITs en test      : {len(test_cuits):,} ({len(test_cuits)/len(cuit_flag)*100:.1f}%)")
    print(f"  Filas train / test : {len(df_train):,} / {len(df_test):,}")

    return df_train, df_test


def main():
    print("=" * 58)
    print("Script 03 - Merge macro + split train/test (Tesis_v2)")
    print("=" * 58)

    if not DATASET_EWS.exists():
        print(f"ERROR: no se encuentra {DATASET_EWS.name}")
        print("Corré primero 02_construir_dataset.py")
        sys.exit(1)

    print(f"\n[1] Cargando {DATASET_EWS.name}...")
    df = pd.read_parquet(DATASET_EWS)
    print(f"  {len(df):,} filas, {df['cuit'].nunique():,} CUITs")

    print(f"\n[2] Merge con macro...")
    df = merge_macro(df)

    print(f"\n[3] Split train/test a nivel CUIT (test_size={TEST_SIZE}, seed={RANDOM_STATE})...")
    df_train, df_test = split_por_cuit(df, TEST_SIZE, RANDOM_STATE)

    print(f"\n[4] Guardando...")
    df_train.to_parquet(OUT_TRAIN, compression="snappy", index=False)
    df_test.to_parquet(OUT_TEST, compression="snappy", index=False)
    print(f"  {OUT_TRAIN.name}: {OUT_TRAIN.stat().st_size / 1024**2:.1f} MB")
    print(f"  {OUT_TEST.name}: {OUT_TEST.stat().st_size / 1024**2:.1f} MB")

    print("\nListo. Siguiente paso: correr el modelado (adaptar 09_modelado_ews.py de la v1,")
    print("o escribir un 04_modelado.py propio de Tesis_v2) sobre dataset_train.parquet /")
    print("dataset_test.parquet.")


if __name__ == "__main__":
    main()

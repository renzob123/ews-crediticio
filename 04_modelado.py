"""
Tesis EWS Crediticio - v2 (Tesis_v2)
Script 04: Modelado

Entrena y evalúa un modelo sobre dataset_train.parquet / dataset_test.parquet
(salidas de 03_split.py). Cada corrida es UNA combinación de:

  --modelo      xgboost (principal) | lightgbm (sensibilidad: otro boosting) |
                logistica (modelo base)
  --target      target_b (principal) | target_a | target_c | target_*_t1 (sensibilidad
                de la ventana t+1, duda 1)
  --validacion  cuit: train = CUITs de train, test = CUITs de test
                oot : además separa por tiempo: train = CUITs de train en períodos
                      oot == "train", test = CUITs de test en períodos oot == "test"
                      (la más conservadora: otras personas y meses posteriores)
  --montos      principal (reales con CER + ratios/deciles) | nominal | real | ratios
                (sensibilidad de la duda 2)
  --macro       agrega las 10 variables macro (sensibilidad; desde 2026-11-25 el
                modelo principal va sin macro porque fuera de tiempo no aporta)

Metodología (docs/decisiones_metodologicas.md, dudas 4 y 6):
  - Se entrena con todas las filas (una por CUIT × período).
  - XGBoost / LightGBM: hiperparámetros elegidos con búsqueda aleatoria y GroupKFold por
    CUIT (todas las filas de un CUIT en el mismo fold), con early stopping en el
    fold de validación. Por costo, la búsqueda se hace sobre una muestra de CUITs
    de train (--frac-tuning); el modelo final se entrena con todo train.
  - Regresión logística: preprocesamiento dentro de un Pipeline ajustado solo
    con train (recorte de extremos, imputación con indicador de faltante,
    escalado, one-hot). No entra en memoria con ~25M filas, así que se entrena
    con una muestra de CUITs de train (--frac-logistica).
  - Desbalance: peso de clase (scale_pos_weight / class_weight="balanced"),
    sin remuestreo. Las métricas usan la prevalencia real.
  - El test se usa una sola vez, al final.
  - Métricas por mes de scoring t (AUC, Gini, PR-AUC; captura, precisión y lift en
    el top 5/10/20%; calibración: Brier y probabilidad media sobre tasa observada),
    reportadas como media ± desvío entre meses, con intervalos de confianza por
    bootstrap por CUIT (Poisson: cada CUIT recibe un peso ~ Poisson(1), que
    equivale a remuestrear CUITs con todas sus filas). Además: métricas sobre
    todas las filas juntas y con una fila al azar por CUIT (robustez).

Salidas en resultados/<corrida>/:
  metricas.json, metricas_por_mes.csv, importancia.csv, tuning.csv (boosting),
  modelo.json / modelo.txt / modelo.joblib (no van al repo).
No se guardan scores por CUIT.

Uso:
  python 04_modelado.py                                   # xgboost, target_b, cuit, principal, sin macro
  python 04_modelado.py --modelo lightgbm
  python 04_modelado.py --modelo logistica
  python 04_modelado.py --validacion oot --macro
  python 04_modelado.py --frac-datos 0.02                 # prueba rápida con 2% de los CUITs
Requiere: pandas, pyarrow, scikit-learn, xgboost, lightgbm, joblib
"""

import argparse
import gc
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

_SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN = _SCRIPT_DIR / "dataset_train.parquet"
TEST  = _SCRIPT_DIR / "dataset_test.parquet"
OUT_DIR = _SCRIPT_DIR / "resultados"

SEED = 42

# ── variables ────────────────────────────────────────────────────────────────────
BASE = [
    "sit_sant_t", "sit_sant_t1", "sit_sant_t2", "sit_sant_t3",
    "meses_consec_normal_sant_6m", "meses_con_deuda_sant_3m", "meses_con_deuda_sant_6m",
    "judicial_sant_t",
    "n_entidades_otras_t", "peor_sit_otras_t", "n_ent_sit2_otras_t", "n_ent_sit3plus_otras_t",
    "tiene_judicial_otras_t",
    "otras_tiene_banco_publico", "otras_tiene_banco_privado", "otras_tiene_banco_digital",
    "otras_tiene_financiera", "otras_tiene_billetera", "otras_tiene_prov_nofin",
    "otras_n_tipos_entidad", "tiene_deuda_otras",
    "n_entidades_total_t", "peor_sit_sistema_t",
    "meses_con_deuda_sistema_3m", "meses_con_deuda_sistema_6m", "deuda_sistema_t5",
]
CATEGORICAS = ["sexo_proxy", "rango_edad_proxy"]
# categorías fijas (las mismas en train, test y en cada bloque de lectura)
CATEGORIAS = {"sexo_proxy": ["F", "I", "M"],
              "rango_edad_proxy": ["18_30", "30_50", "50_65", "65_mas"]}
MONTOS_NOMINAL = ["monto_sant_t", "monto_sant_t3", "var_monto_3m_sant", "log_monto_sant",
                  "monto_otras_t", "monto_otras_t3", "log_monto_otras"]
MONTOS_REAL = ["monto_sant_t_real", "monto_sant_t3_real", "var_monto_3m_sant_real",
               "log_monto_sant_real", "monto_otras_t_real", "monto_otras_t3_real",
               "log_monto_otras_real"]
RATIOS = ["share_sant_t", "share_sant_t3", "dec_monto_sant_t", "dec_monto_sant_t3",
          "dec_monto_otras_t", "dec_monto_otras_t3"]
CONJUNTOS_MONTOS = {
    "principal": MONTOS_REAL + RATIOS,
    "nominal":   MONTOS_NOMINAL,
    "real":      MONTOS_REAL,
    "ratios":    RATIOS,
}
MACRO = ["mora_3plus_privado_pct", "sit2_privado_pct", "var_mora3plus_3m", "var_sit2_3m",
         "tasa_consumo_corto_pct", "badlar_privados_tna_pct", "spread_consumo_badlar_pct",
         "var_tc_3m_pct", "pct_personales", "var_credito_real_3m_pct"]

# para la logística: los montos en niveles van en log (sus columnas log_* se
# descartan para no duplicar) y las variaciones % se recortan en los extremos
NIVELES_MONTO = {"monto_sant_t", "monto_sant_t3", "monto_otras_t", "monto_otras_t3",
                 "monto_sant_t_real", "monto_sant_t3_real", "monto_otras_t_real",
                 "monto_otras_t3_real"}
LOGS_MONTO = {"log_monto_sant", "log_monto_otras", "log_monto_sant_real", "log_monto_otras_real"}

# ── búsqueda de hiperparámetros ──────────────────────────────────────────────────
# Espacios equivalentes entre los dos boostings (profundidad/hojas, tasa de
# aprendizaje, mínimo por hoja, submuestreo de columnas y filas, regularización L2)
ESPACIO_XGB = {
    "max_depth":        [4, 6, 8, 10],
    "learning_rate":    [0.05, 0.1],
    "min_child_weight": [1, 10, 50],     # en unidades de hessiano, no de filas
    "colsample_bytree": [0.6, 0.8, 1.0],
    "subsample":        [0.7, 1.0],
    "reg_lambda":       [1.0, 10.0],
}
ESPACIO_LGB = {
    "num_leaves":        [31, 63, 127, 255],
    "learning_rate":     [0.05, 0.1],
    "min_child_samples": [100, 500, 2000],
    "feature_fraction":  [0.6, 0.8, 1.0],
    "bagging_fraction":  [0.7, 1.0],
    "lambda_l2":         [0.0, 1.0, 10.0],
}
ESPACIOS = {"xgboost": ESPACIO_XGB, "lightgbm": ESPACIO_LGB}
ENTEROS = {"num_leaves", "min_child_samples", "max_depth"}
MAX_ARBOLES = 3000
PACIENCIA = 50
CAPTURAS = (0.05, 0.10, 0.20)


# ── carga ─────────────────────────────────────────────────────────────────────────
MODULOS = {"datos": 9973, "tuning": 7919}   # primos distintos → muestras independientes

def muestra_cuits(cuit: np.ndarray, frac: float, uso: str = "datos") -> np.ndarray:
    """
    Máscara determinística por CUIT (todas las filas de un CUIT entran o no).
    Cada uso tiene su propio módulo para que la muestra de tuning no quede
    anidada en la de datos.
    """
    if frac >= 1:
        return np.ones(len(cuit), dtype=bool)
    mod = MODULOS[uso]
    return ((cuit // 10) % mod) < frac * mod    # // 10 descarta el dígito verificador


def lotes_filtrados(path: Path, leer: list, target: str, oot_col: str | None,
                    oot_valor: str | None, frac: float, frac_tuning: float | None = None):
    """
    Recorre el parquet de a un bloque (sin lectura anticipada) y devuelve cada
    bloque como tabla de Arrow con solo las filas con el target válido, en el
    tramo OOT pedido y en la fracción de CUITs pedida (y, si se pide, solo la
    muestra de tuning). El CUIT pasa a entero para no cargar millones de strings.
    """
    escaner = ds.dataset(path).scanner(columns=leer, filter=ds.field(target).is_valid(),
                                       batch_size=500_000, batch_readahead=1,
                                       fragment_readahead=1)
    for lote in escaner.to_batches():
        if lote.num_rows == 0:
            continue
        cuit = pc.cast(lote.column("cuit"), pa.int64()).to_numpy()
        keep = muestra_cuits(cuit, frac)
        if frac_tuning is not None:
            keep &= muestra_cuits(cuit, frac_tuning, "tuning")
        if oot_col:
            keep &= pc.equal(pc.cast(lote.column(oot_col), pa.string()), oot_valor) \
                      .to_numpy(zero_copy_only=False)
        if not keep.any():
            continue
        tabla = pa.Table.from_batches([lote]).filter(pa.array(keep))
        tabla = tabla.set_column(0, "cuit", pc.cast(tabla.column("cuit"), pa.int64()))
        if oot_col:
            tabla = tabla.drop_columns([oot_col])
        del lote
        yield tabla


def fijar_tipos(df: pd.DataFrame, target: str) -> pd.DataFrame:
    for c in CATEGORICAS:
        if c in df:
            nulos = df[c].isna().sum()
            df[c] = df[c].astype(pd.CategoricalDtype(CATEGORIAS[c]))
            if df[c].isna().sum() != nulos:
                raise ValueError(f"{c} tiene valores fuera de CATEGORIAS")
    df[target] = df[target].astype("int8")
    return df


def a_pandas(tabla: pa.Table) -> pd.DataFrame:
    # sin los metadatos de pandas del parquet, que volverían a convertir cuit en texto
    return tabla.replace_schema_metadata(None).to_pandas()


def cargar(path: Path, columnas: list, target: str, oot_col: str | None,
           oot_valor: str | None, frac: float, frac_tuning: float | None = None) -> pd.DataFrame:
    """
    Carga en un DataFrame los bloques de lotes_filtrados. Se arma columna por
    columna, liberando los bloques de Arrow de cada columna a medida que pasa a
    pandas, para no tener dos copias completas de los datos.
    """
    leer = ["cuit", "periodo", target, *columnas] + ([oot_col] if oot_col else [])
    partes = {}   # columna → bloques de Arrow ya filtrados
    for tabla in lotes_filtrados(path, leer, target, oot_col, oot_valor, frac, frac_tuning):
        for c in tabla.column_names:
            partes.setdefault(c, []).extend(tabla.column(c).chunks)
        del tabla
    if not partes:
        return pd.DataFrame(columns=leer)
    columnas_df = {}
    for c in list(partes):
        trozos = partes.pop(c)
        columnas_df[c] = pa.chunked_array(trozos).to_pandas()
        del trozos
    df = pd.DataFrame(columnas_df, copy=False)
    del columnas_df
    return fijar_tipos(df, target)


class IterTrain(xgb.DataIter):
    """
    Entrega train a XGBoost de a un bloque del parquet para armar una matriz en
    memoria externa (ExtMemQuantileDMatrix): XGBoost guarda las páginas ya
    discretizadas en disco (cache_dir) y las lee durante el entrenamiento.
    Con todo train el pico es ~3,4 GB; armar la matriz en memoria desde un
    DataFrame superaba los 8 GB. En la primera pasada junta los datos
    descriptivos de train (filas, CUITs, períodos, positivos).
    """
    def __init__(self, path, features, target, oot_col, oot_valor, frac, cache_dir):
        self.features, self.target = features, target
        self.leer = ["cuit", "periodo", target, *features] + ([oot_col] if oot_col else [])
        self.args = (path, self.leer, target, oot_col, oot_valor, frac)
        self._gen = None
        self.pasadas = 0
        self._cuits, self._pos, self._filas, self._per = [], 0, 0, [np.inf, -np.inf]
        self.stats = None
        super().__init__(cache_prefix=str(Path(cache_dir) / "train"), release_data=True,
                         on_host=False)

    def reset(self):
        if self._gen is not None and self.stats is None:
            self._cerrar_stats()
        self._gen = None

    def _cerrar_stats(self):
        cuits = np.unique(np.concatenate(self._cuits)) if self._cuits else np.array([])
        self.stats = {"filas": int(self._filas), "cuits": int(len(cuits)),
                      "periodos": [int(self._per[0]), int(self._per[1])],
                      "positivos": int(self._pos),
                      "prevalencia": float(self._pos / self._filas) if self._filas else float("nan")}
        self._cuits = []

    def next(self, input_data):
        if self._gen is None:
            self._gen = lotes_filtrados(*self.args)
            self.pasadas += 1
        try:
            tabla = next(self._gen)
        except StopIteration:
            return False
        df = fijar_tipos(a_pandas(tabla), self.target)
        del tabla
        y = df[self.target].to_numpy()
        if self.stats is None:
            self._cuits.append(np.unique(df["cuit"].to_numpy()))
            self._pos += int(y.sum()); self._filas += len(y)
            self._per = [min(self._per[0], df["periodo"].min()), max(self._per[1], df["periodo"].max())]
        input_data(data=df[self.features], label=y)
        return True


def a_matriz(df: pd.DataFrame, features: list) -> np.ndarray:
    """Variables como matriz float32 (categóricas como su código; faltantes = NaN)."""
    X = np.empty((len(df), len(features)), dtype=np.float32)
    for j, c in enumerate(features):
        if c in CATEGORICAS:
            cod = df[c].cat.codes.to_numpy().astype(np.float32)
            cod[cod < 0] = np.nan
            X[:, j] = cod
        else:
            X[:, j] = df[c].to_numpy(dtype=np.float32, na_value=np.nan)
    return X


def train_en_disco(path, features, target, oot_col, oot_valor, frac, archivo):
    """
    Escribe las variables de train de a bloques en un archivo binario float32 y
    lo devuelve abierto como memory-map, junto con el target y los datos
    descriptivos de train. El sistema operativo lee el archivo a demanda, así
    que todo train nunca ocupa RAM a la vez (para LightGBM con todo train).
    """
    leer = ["cuit", "periodo", target, *features] + ([oot_col] if oot_col else [])
    ys, cuits, filas, per = [], [], 0, [np.inf, -np.inf]
    with open(archivo, "wb") as f:
        for tabla in lotes_filtrados(path, leer, target, oot_col, oot_valor, frac):
            df = fijar_tipos(a_pandas(tabla), target)
            del tabla
            f.write(a_matriz(df, features).tobytes())
            ys.append(df[target].to_numpy())
            cuits.append(np.unique(df["cuit"].to_numpy()))
            filas += len(df)
            per = [min(per[0], df["periodo"].min()), max(per[1], df["periodo"].max())]
            del df
    y = np.concatenate(ys) if ys else np.array([], dtype=np.int8)
    stats = {"filas": int(filas), "cuits": int(len(np.unique(np.concatenate(cuits)))) if cuits else 0,
             "periodos": [int(per[0]), int(per[1])] if filas else [0, 0],
             "positivos": int(y.sum()),
             "prevalencia": float(y.mean()) if filas else float("nan")}
    X = np.memmap(archivo, dtype=np.float32, mode="r", shape=(filas, len(features))) if filas else None
    return X, y, stats


# ── métricas ──────────────────────────────────────────────────────────────────────
def capturas(y, p, w):
    """
    Para el top X% de scores (ponderado): captura (% de los positivos que caen
    ahí), precisión (% de positivos dentro del top) y lift (precisión sobre la
    prevalencia; 1 = elegir al azar).
    """
    orden = np.argsort(-p, kind="stable")
    w_ord, pos_ord = w[orden], (y[orden] * w[orden])
    cum_w, cum_pos = np.cumsum(w_ord), np.cumsum(pos_ord)
    total_w, total_pos = cum_w[-1], cum_pos[-1]
    out = {}
    for x in CAPTURAS:
        j = min(np.searchsorted(cum_w, x * total_w, side="right"), len(cum_pos) - 1)
        pct = int(x * 100)
        if total_pos > 0:
            out[f"captura_top{pct}"] = float(cum_pos[j] / total_pos)
            out[f"precision_top{pct}"] = float(cum_pos[j] / cum_w[j])
            out[f"lift_top{pct}"] = float((cum_pos[j] / cum_w[j]) / (total_pos / total_w))
        else:
            out[f"captura_top{pct}"] = out[f"precision_top{pct}"] = out[f"lift_top{pct}"] = np.nan
    return out


def metricas(y, p, w=None):
    """
    Discriminación: AUC, Gini (= 2·AUC − 1, la forma usual en riesgo de crédito),
    PR-AUC y captura / precisión / lift en el top. Calibración: Brier y cociente
    entre la probabilidad media predicha y la tasa observada (1 = calibrado; con
    scale_pos_weight los boostings dan probabilidades infladas, > 1).
    """
    if w is None:
        w = np.ones(len(y))
    if y.min() == y.max():
        return {}
    prev = float(np.average(y, weights=w))
    auc = roc_auc_score(y, p, sample_weight=w)
    m = {"prevalencia": prev, "auc": auc, "gini": 2 * auc - 1,
         "pr_auc": average_precision_score(y, p, sample_weight=w)}
    m.update(capturas(y, p, w))
    m["brier"] = float(np.average((p - y) ** 2, weights=w))
    m["p_media_sobre_tasa"] = float(np.average(p, weights=w) / prev)
    return m


def metricas_por_mes(y, p, periodo, w=None):
    filas = []
    for per in np.unique(periodo):
        i = periodo == per
        ww = None if w is None else w[i]
        fila = {"periodo": int(per), "n": int(i.sum())}
        fila.update(metricas(y[i], p[i], ww))
        filas.append(fila)
    return pd.DataFrame(filas)


def bootstrap_cuit(y, p, periodo, cuit, n_boot):
    """IC 95% de la media entre meses, con pesos Poisson(1) por CUIT."""
    rng = np.random.default_rng(SEED)
    codigos, unicos = pd.factorize(cuit)
    medias = []
    for b in range(n_boot):
        w = rng.poisson(1.0, len(unicos)).astype(float)[codigos]
        mm = metricas_por_mes(y, p, periodo, w)
        medias.append(mm.drop(columns=["periodo", "n", "prevalencia"]).mean())
        if (b + 1) % 20 == 0:
            print(f"    bootstrap {b + 1}/{n_boot}")
    medias = pd.DataFrame(medias)
    return {c: [float(medias[c].quantile(0.025)), float(medias[c].quantile(0.975))]
            for c in medias.columns}


# ── modelos ───────────────────────────────────────────────────────────────────────
class Recorte(BaseEstimator, TransformerMixin):
    """Recorta cada columna a sus percentiles [bajo, alto] calculados en train."""
    def __init__(self, bajo=0.5, alto=99.5):
        self.bajo, self.alto = bajo, alto

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lim_ = np.nanpercentile(X, [self.bajo, self.alto], axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lim_[0], self.lim_[1])

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def log_no_negativo(X):
    return np.log1p(np.clip(np.asarray(X, dtype=float), 0, None))


def pipeline_logistica(features):
    niveles = [c for c in features if c in NIVELES_MONTO]
    numericas = [c for c in features
                 if c not in NIVELES_MONTO and c not in LOGS_MONTO and c not in CATEGORICAS]
    cats = [c for c in features if c in CATEGORICAS]
    imputar = lambda: SimpleImputer(strategy="median", add_indicator=True)
    pre = ColumnTransformer([
        ("num", Pipeline([("recorte", Recorte()), ("imp", imputar()),
                          ("esc", StandardScaler())]), numericas),
        ("montos", Pipeline([("log", FunctionTransformer(log_no_negativo, feature_names_out="one-to-one")), ("imp", imputar()),
                             ("esc", StandardScaler())]), niveles),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cats),
    ])
    return Pipeline([("pre", pre),
                     ("lr", LogisticRegression(class_weight="balanced", max_iter=500))])


def peso_positivos(y):
    pos = y.sum()
    return float((len(y) - pos) / pos)


def params_boosting(modelo, cfg, spw):
    if modelo == "xgboost":
        return {"objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist",
                "seed": SEED, "scale_pos_weight": spw, "verbosity": 0, **cfg}
    return {"objective": "binary", "metric": "average_precision", "verbosity": -1,
            "seed": SEED, "bagging_freq": 1, "scale_pos_weight": spw, **cfg}


def ajustar_cv(modelo, cfg, X_tr, y_tr, X_va, y_va):
    """Entrena con early stopping en el fold de validación: (PR-AUC, árboles)."""
    params = params_boosting(modelo, cfg, peso_positivos(y_tr))
    if modelo == "xgboost":
        dtr = xgb.QuantileDMatrix(X_tr, y_tr, enable_categorical=True)
        dva = xgb.QuantileDMatrix(X_va, y_va, enable_categorical=True, ref=dtr)
        m = xgb.train(params, dtr, MAX_ARBOLES, evals=[(dva, "valid")],
                      early_stopping_rounds=PACIENCIA, verbose_eval=False)
        return m.best_score, m.best_iteration + 1
    dtr = lgb.Dataset(X_tr, y_tr)
    dva = lgb.Dataset(X_va, y_va, reference=dtr)
    m = lgb.train(params, dtr, MAX_ARBOLES, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(PACIENCIA, verbose=False)])
    return m.best_score["valid_0"]["average_precision"], m.best_iteration


def tuning_boosting(modelo, X, y, grupos, n_configs, n_folds):
    rng = np.random.default_rng(SEED)
    configs = [{k: v[rng.integers(len(v))] for k, v in ESPACIOS[modelo].items()}
               for _ in range(n_configs)]
    folds = list(GroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
                 .split(X, y, groups=grupos))
    filas = []
    for i, cfg in enumerate(configs, 1):
        aps, iters = [], []
        for tr, va in folds:
            ap_, it = ajustar_cv(modelo, cfg, X.iloc[tr], y[tr], X.iloc[va], y[va])
            aps.append(ap_)
            iters.append(it)
        filas.append({**cfg, "pr_auc_cv": float(np.mean(aps)), "pr_auc_cv_sd": float(np.std(aps)),
                      "arboles": int(np.mean(iters))})
        print(f"  config {i}/{n_configs}: PR-AUC {np.mean(aps):.4f} ± {np.std(aps):.4f}"
              f"  ({int(np.mean(iters))} árboles)  {cfg}")
    return pd.DataFrame(filas).sort_values("pr_auc_cv", ascending=False)


# ── main ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", choices=["xgboost", "lightgbm", "logistica"], default="xgboost")
    ap.add_argument("--target", default="target_b",
                    choices=[f"target_{t}{s}" for s in ("", "_t1") for t in "abc"])
    ap.add_argument("--validacion", choices=["cuit", "oot"], default="cuit")
    ap.add_argument("--montos", choices=list(CONJUNTOS_MONTOS), default="principal")
    ap.add_argument("--macro", action="store_true", help="agregar las variables macro")
    ap.add_argument("--frac-datos", type=float, default=1.0,
                    help="fracción de CUITs a usar en train y test (para pruebas rápidas)")
    ap.add_argument("--frac-tuning", type=float, default=0.2,
                    help="fracción de los CUITs de train para la búsqueda de hiperparámetros")
    ap.add_argument("--frac-logistica", type=float, default=0.25,
                    help="fracción de los CUITs de train para entrenar la logística")
    ap.add_argument("--configs", type=int, default=6, help="configuraciones a probar (boosting)")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=100)
    ap.add_argument("--reusar-tuning", action="store_true",
                    help="usar el tuning.csv ya guardado de esta corrida en vez de repetir la búsqueda")
    ap.add_argument("--solo-evaluar", action="store_true",
                    help="no entrenar: cargar el modelo ya guardado de esta corrida y recalcular "
                         "las métricas de test (por ejemplo, al agregar métricas nuevas)")
    args = ap.parse_args()

    t0 = time.time()
    features = BASE + CONJUNTOS_MONTOS[args.montos] + (MACRO if args.macro else []) + CATEGORICAS
    oot_col = "oot_" + args.target.removeprefix("target_") if args.validacion == "oot" else None
    nombre = (f"{args.modelo}_{args.target}_{args.validacion}_{args.montos}_"
              f"{'macro' if args.macro else 'sinmacro'}"
              + (f"_prueba{args.frac_datos:g}" if args.frac_datos < 1 else ""))
    out = OUT_DIR / nombre
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(f"Script 04 - Modelado: {nombre}")
    print("=" * 64)
    print(f"  {len(features)} variables")

    boosting = args.modelo in ("xgboost", "lightgbm")

    def cargar_train(frac, frac_tuning=None):
        df = cargar(TRAIN, features, args.target, oot_col, "train", frac, frac_tuning)
        if df.empty:
            print(f"ERROR: no hay filas de train para {args.target} con validación {args.validacion}.")
            sys.exit(1)
        y = df[args.target].to_numpy()
        print(f"  {len(df):,} filas, {df['cuit'].nunique():,} CUITs, "
              f"períodos {df['periodo'].min()}-{df['periodo'].max()}, positivos {y.mean()*100:.2f}%")
        return df, y

    resultado = {"corrida": nombre, "args": vars(args), "features": features}

    if args.solo_evaluar:
        # se conservan train, hiperparámetros y tiempos de la corrida original
        previo = json.loads((out / "metricas.json").read_text(encoding="utf-8"))
        if previo["features"] != features:
            raise ValueError("las variables no coinciden con las del modelo guardado")
        print(f"\n[1-4] Cargando el modelo guardado de {out} (sin reentrenar)")
        if args.modelo == "xgboost":
            modelo = xgb.Booster(model_file=str(out / "modelo.json"))
        elif args.modelo == "lightgbm":
            modelo = lgb.Booster(model_file=str(out / "modelo.txt"))
        else:
            modelo = joblib.load(out / "modelo.joblib")
        evaluar(args, features, oot_col, out, funcion_prediccion(args.modelo, modelo, features),
                {**previo, "reevaluado": time.strftime("%Y-%m-%d %H:%M")}, t0)
        return

    # ── búsqueda de hiperparámetros (solo boosting), con su muestra de CUITs ──
    # Se carga solo la muestra de tuning y se libera antes de cargar todo train,
    # para no tener las dos cosas en memoria a la vez.
    if boosting and args.reusar_tuning and (out / "tuning.csv").exists():
        print(f"\n[1-2] Reutilizando la búsqueda de hiperparámetros guardada en {out / 'tuning.csv'}")
        tuning = pd.read_csv(out / "tuning.csv")
    elif boosting:
        print(f"\n[1] Cargando la muestra de tuning ({args.frac_tuning:g} de los CUITs de train)...")
        sub, y_sub = cargar_train(args.frac_datos, args.frac_tuning)
        print(f"\n[2] Búsqueda de hiperparámetros: {args.configs} configs × GroupKFold({args.folds}) "
              f"por CUIT...")
        grupos = sub["cuit"].to_numpy()
        sub = sub.drop(columns=["cuit", "periodo", args.target])   # sin copia (copy-on-write)
        tuning = tuning_boosting(args.modelo, sub, y_sub, grupos, args.configs, args.folds)
        del sub, y_sub; gc.collect()
        tuning.to_csv(out / "tuning.csv", index=False)
    if boosting:
        mejor = tuning.sort_values("pr_auc_cv", ascending=False).iloc[0].to_dict()
        arboles = int(mejor.pop("arboles"))
        mejor.pop("pr_auc_cv"); mejor.pop("pr_auc_cv_sd")
        mejor = {k: (int(v) if k in ENTEROS else float(v)) for k, v in mejor.items()}
        print(f"  Mejor: {mejor}, {arboles} árboles")

    # ── train ──
    if args.modelo == "xgboost":
        # XGBoost lee train de a bloques directo al QuantileDMatrix (ver IterTrain)
        print(f"\n[3] Armando la matriz de train en memoria externa (fracción de CUITs "
              f"{args.frac_datos:g})...")
        cache_dir = tempfile.mkdtemp(prefix="xgb_cache_")   # se borra al terminar de entrenar
        it = IterTrain(TRAIN, features, args.target, oot_col, "train" if oot_col else None,
                       args.frac_datos, cache_dir)
        dtr = xgb.ExtMemQuantileDMatrix(it, enable_categorical=True)
        if it.stats is None:
            it._cerrar_stats()
        st = it.stats
        del it; gc.collect()
        if st["filas"] == 0:
            print(f"ERROR: no hay filas de train para {args.target} con validación {args.validacion}.")
            sys.exit(1)
        print(f"  {st['filas']:,} filas, {st['cuits']:,} CUITs, períodos {st['periodos'][0]}-"
              f"{st['periodos'][1]}, positivos {st['prevalencia']*100:.2f}%")
        resultado["train"] = {k: st[k] for k in ("filas", "cuits", "periodos", "prevalencia")}
        spw = (st["filas"] - st["positivos"]) / st["positivos"]
    elif args.modelo == "lightgbm":
        # LightGBM arma su Dataset desde un memory-map en disco (ver train_en_disco)
        print(f"\n[3] Escribiendo train a disco por bloques (fracción de CUITs {args.frac_datos:g})...")
        cache_dir = tempfile.mkdtemp(prefix="lgb_cache_")   # se borra al terminar de entrenar
        X_tr, y_tr, st = train_en_disco(TRAIN, features, args.target, oot_col,
                                        "train" if oot_col else None, args.frac_datos,
                                        Path(cache_dir) / "train.f32")
        if st["filas"] == 0:
            print(f"ERROR: no hay filas de train para {args.target} con validación {args.validacion}.")
            sys.exit(1)
        print(f"  {st['filas']:,} filas, {st['cuits']:,} CUITs, períodos {st['periodos'][0]}-"
              f"{st['periodos'][1]}, positivos {st['prevalencia']*100:.2f}%")
        resultado["train"] = {k: st[k] for k in ("filas", "cuits", "periodos", "prevalencia")}
        spw = (st["filas"] - st["positivos"]) / st["positivos"]
    else:
        frac_train = args.frac_datos * (1.0 if boosting else args.frac_logistica)
        print(f"\n[{3 if boosting else 1}] Cargando train (fracción de CUITs {frac_train:g})...")
        tr, y_tr = cargar_train(frac_train)
        resultado["train"] = {"filas": len(tr), "cuits": int(tr["cuit"].nunique()),
                              "periodos": [int(tr["periodo"].min()), int(tr["periodo"].max())],
                              "prevalencia": float(y_tr.mean())}
        tr = tr.drop(columns=["cuit", "periodo", args.target])   # quedan solo las variables
        spw = peso_positivos(y_tr)

    # ── entrenamiento ──
    if boosting:
        print(f"\n[4] Entrenando el modelo final con todo train...")
        params = params_boosting(args.modelo, mejor, spw)
        if args.modelo == "xgboost":
            try:
                modelo = xgb.train(params, dtr, arboles)
            finally:
                del dtr; gc.collect()
                shutil.rmtree(cache_dir, ignore_errors=True)
            modelo.save_model(str(out / "modelo.json"))
            gan = modelo.get_score(importance_type="total_gain")
            spl = modelo.get_score(importance_type="weight")
            imp = pd.DataFrame({"variable": features,
                                "ganancia": [gan.get(f, 0.0) for f in features],
                                "splits": [spl.get(f, 0) for f in features]})
        else:
            try:
                dtr = lgb.Dataset(X_tr, y_tr, feature_name=features,
                                  categorical_feature=[c for c in features if c in CATEGORICAS],
                                  free_raw_data=True)
                modelo = lgb.train(params, dtr, arboles)
            finally:
                dtr = X_tr = None; gc.collect()   # cerrar el memory-map antes de borrar
                shutil.rmtree(cache_dir, ignore_errors=True)
            modelo.save_model(str(out / "modelo.txt"))
            imp = pd.DataFrame({"variable": modelo.feature_name(),
                                "ganancia": modelo.feature_importance("gain"),
                                "splits": modelo.feature_importance("split")})
        imp["ganancia_pct"] = imp["ganancia"] / imp["ganancia"].sum() * 100
        imp.sort_values("ganancia", ascending=False).to_csv(out / "importancia.csv", index=False)
        resultado["hiperparametros"] = {**mejor, "arboles": arboles}
    else:
        print(f"\n[2] Entrenando regresión logística (Pipeline ajustado solo con train)...")
        modelo = pipeline_logistica(features)
        modelo.fit(tr, y_tr)
        del tr; gc.collect()
        joblib.dump(modelo, out / "modelo.joblib")
        nombres = modelo.named_steps["pre"].get_feature_names_out()
        coef = pd.DataFrame({"variable": nombres, "coeficiente": modelo.named_steps["lr"].coef_[0]})
        coef["abs"] = coef["coeficiente"].abs()
        coef.sort_values("abs", ascending=False).drop(columns="abs") \
            .to_csv(out / "importancia.csv", index=False)

    evaluar(args, features, oot_col, out, funcion_prediccion(args.modelo, modelo, features),
            resultado, t0)


def funcion_prediccion(tipo, modelo, features):
    if tipo == "xgboost":
        return lambda X: modelo.predict(xgb.DMatrix(X, enable_categorical=True))
    if tipo == "lightgbm":
        # de a tramos, con la misma codificación que en el entrenamiento
        return lambda X: np.concatenate(
            [modelo.predict(a_matriz(X.iloc[i:i + 1_000_000], features))
             for i in range(0, len(X), 1_000_000)])
    # de a tramos: el pipeline pasa a float64 y copia los datos (con todo test no entra en memoria)
    return lambda X: np.concatenate(
        [modelo.predict_proba(X.iloc[i:i + 500_000])[:, 1] for i in range(0, len(X), 500_000)])


def evaluar(args, features, oot_col, out, predecir, resultado, t0):
    # ── test (una sola vez) ──
    print(f"\n[5] Evaluando en test...")
    te = cargar(TEST, features, args.target, oot_col, "test", args.frac_datos)
    y_te, per_te, cuit_te = te[args.target].to_numpy(), te["periodo"].to_numpy(), te["cuit"].to_numpy()
    p_te = predecir(te[features])
    del te; gc.collect()
    print(f"  {len(y_te):,} filas, {len(np.unique(cuit_te)):,} CUITs, "
          f"{len(np.unique(per_te))} meses, positivos {y_te.mean()*100:.2f}%")

    por_mes = metricas_por_mes(y_te, p_te, per_te)
    por_mes.to_csv(out / "metricas_por_mes.csv", index=False)
    cols_m = [c for c in por_mes.columns if c not in ("periodo", "n", "prevalencia")]

    rng = np.random.default_rng(SEED)
    una_fila = pd.DataFrame({"cuit": cuit_te, "r": rng.random(len(cuit_te))}) \
        .groupby("cuit")["r"].idxmin().to_numpy()

    print(f"\n[6] Bootstrap por CUIT ({args.bootstrap} réplicas)...")
    ic = bootstrap_cuit(y_te, p_te, per_te, cuit_te, args.bootstrap) if args.bootstrap else {}

    resultado["test"] = {
        "filas": int(len(y_te)), "cuits": int(len(np.unique(cuit_te))),
        "meses": [int(p) for p in np.unique(per_te)], "prevalencia": float(y_te.mean()),
        "por_mes_media": {c: float(por_mes[c].mean()) for c in cols_m},
        "por_mes_sd": {c: float(por_mes[c].std()) for c in cols_m},
        "por_mes_ic95_bootstrap_cuit": ic,
        "todas_las_filas": metricas(y_te, p_te),
        "una_fila_por_cuit": metricas(y_te[una_fila], p_te[una_fila]),
    }
    # al reevaluar se conserva el tiempo de la corrida original
    clave_t = "minutos_reevaluacion" if "reevaluado" in resultado else "minutos"
    resultado[clave_t] = round((time.time() - t0) / 60, 1)
    with open(out / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    # ── resumen ──
    r = resultado["test"]
    print(f"\n  Métricas por mes de scoring (media ± desvío entre {len(por_mes)} meses"
          f"{', IC 95% bootstrap por CUIT' if ic else ''}):")
    for c in cols_m:
        extra = f"  IC [{ic[c][0]:.4f}, {ic[c][1]:.4f}]" if c in ic else ""
        print(f"    {c:<18} {r['por_mes_media'][c]:.4f} ± {r['por_mes_sd'][c]:.4f}{extra}")
    for etiqueta, clave in (("Todas las filas juntas", "todas_las_filas"),
                            ("Una fila por CUIT", "una_fila_por_cuit")):
        mm = r[clave]
        print(f"  {etiqueta:<23}: AUC {mm['auc']:.4f}  PR-AUC {mm['pr_auc']:.4f}  "
              f"(prevalencia {mm['prevalencia']*100:.2f}%)")
    print(f"\n  Resultados en {out}  ({resultado[clave_t]} min)")


if __name__ == "__main__":
    main()

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
  --sin-macro   sin las 10 variables macro (duda 5: con vs sin macro)

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
  - Métricas por mes de scoring t (AUC, PR-AUC, captura en el top 5/10/20%),
    reportadas como media ± desvío entre meses, con intervalos de confianza por
    bootstrap por CUIT (Poisson: cada CUIT recibe un peso ~ Poisson(1), que
    equivale a remuestrear CUITs con todas sus filas). Además: métricas sobre
    todas las filas juntas y con una fila al azar por CUIT (robustez).

Salidas en resultados/<corrida>/:
  metricas.json, metricas_por_mes.csv, importancia.csv, tuning.csv (boosting),
  modelo.json / modelo.txt / modelo.joblib (no van al repo).
No se guardan scores por CUIT.

Uso:
  python 04_modelado.py                                   # xgboost, target_b, cuit, principal, con macro
  python 04_modelado.py --modelo lightgbm
  python 04_modelado.py --modelo logistica
  python 04_modelado.py --validacion oot --sin-macro
  python 04_modelado.py --frac-datos 0.02                 # prueba rápida con 2% de los CUITs
Requiere: pandas, pyarrow, scikit-learn, xgboost, lightgbm, joblib
"""

import argparse
import gc
import json
import sys
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


def cargar(path: Path, columnas: list, target: str, oot_col: str | None,
           oot_valor: str | None, frac: float) -> pd.DataFrame:
    """
    Lee por bloques solo las columnas necesarias, las filas con el target válido
    (y en el tramo OOT pedido) y la fracción de CUITs pedida. El CUIT pasa a
    entero para no cargar millones de strings.
    """
    leer = ["cuit", "periodo", target, *columnas] + ([oot_col] if oot_col else [])
    escaner = ds.dataset(path).scanner(columns=leer, filter=ds.field(target).is_valid(),
                                       batch_size=1_000_000)
    partes = []
    for lote in escaner.to_batches():
        if lote.num_rows == 0:
            continue
        cuit = pc.cast(lote.column("cuit"), pa.int64()).to_numpy()
        keep = muestra_cuits(cuit, frac)
        if oot_col:
            keep &= pc.equal(pc.cast(lote.column(oot_col), pa.string()), oot_valor) \
                      .to_numpy(zero_copy_only=False)
        if not keep.any():
            continue
        tabla = pa.Table.from_batches([lote]).filter(pa.array(keep))
        tabla = tabla.set_column(0, "cuit", pc.cast(tabla.column("cuit"), pa.int64()))
        if oot_col:
            tabla = tabla.drop_columns([oot_col])
        partes.append(tabla)
    if not partes:
        return pd.DataFrame(columns=leer)
    # sin los metadatos de pandas del parquet, que volverían a convertir cuit en texto
    df = pa.concat_tables(partes).replace_schema_metadata(None).to_pandas()
    for c in CATEGORICAS:
        if c in df:
            df[c] = df[c].astype("category")
    df[target] = df[target].astype("int8")
    return df


# ── métricas ──────────────────────────────────────────────────────────────────────
def capturas(y, p, w):
    """% de los positivos (ponderados) que caen en el top X% de scores."""
    orden = np.argsort(-p, kind="stable")
    w_ord, pos_ord = w[orden], (y[orden] * w[orden])
    cum_w, cum_pos = np.cumsum(w_ord), np.cumsum(pos_ord)
    total_w, total_pos = cum_w[-1], cum_pos[-1]
    out = {}
    for x in CAPTURAS:
        k = np.searchsorted(cum_w, x * total_w, side="right")
        out[f"captura_top{int(x*100)}"] = float(cum_pos[min(k, len(cum_pos) - 1)] / total_pos) \
            if total_pos > 0 else np.nan
    return out


def metricas(y, p, w=None):
    if w is None:
        w = np.ones(len(y))
    if y.min() == y.max():
        return {}
    m = {"prevalencia": float(np.average(y, weights=w)),
         "auc": roc_auc_score(y, p, sample_weight=w),
         "pr_auc": average_precision_score(y, p, sample_weight=w)}
    m.update(capturas(y, p, w))
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
    ap.add_argument("--sin-macro", action="store_true")
    ap.add_argument("--frac-datos", type=float, default=1.0,
                    help="fracción de CUITs a usar en train y test (para pruebas rápidas)")
    ap.add_argument("--frac-tuning", type=float, default=0.2,
                    help="fracción de los CUITs de train para la búsqueda de hiperparámetros")
    ap.add_argument("--frac-logistica", type=float, default=0.25,
                    help="fracción de los CUITs de train para entrenar la logística")
    ap.add_argument("--configs", type=int, default=10, help="configuraciones a probar (boosting)")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()
    features = BASE + CONJUNTOS_MONTOS[args.montos] + ([] if args.sin_macro else MACRO) + CATEGORICAS
    oot_col = "oot_" + args.target.removeprefix("target_") if args.validacion == "oot" else None
    nombre = (f"{args.modelo}_{args.target}_{args.validacion}_{args.montos}_"
              f"{'sinmacro' if args.sin_macro else 'macro'}"
              + (f"_prueba{args.frac_datos:g}" if args.frac_datos < 1 else ""))
    out = OUT_DIR / nombre
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(f"Script 04 - Modelado: {nombre}")
    print("=" * 64)
    print(f"  {len(features)} variables")

    # ── train ──
    frac_train = args.frac_datos * (args.frac_logistica if args.modelo == "logistica" else 1.0)
    print(f"\n[1] Cargando train (fracción de CUITs {frac_train:g})...")
    tr = cargar(TRAIN, features, args.target, oot_col, "train", frac_train)
    if tr.empty:
        print(f"ERROR: no hay filas de train para {args.target} con validación {args.validacion}.")
        sys.exit(1)
    y_tr = tr[args.target].to_numpy()
    print(f"  {len(tr):,} filas, {tr['cuit'].nunique():,} CUITs, "
          f"períodos {tr['periodo'].min()}-{tr['periodo'].max()}, positivos {y_tr.mean()*100:.2f}%")

    resultado = {"corrida": nombre, "args": vars(args), "features": features,
                 "train": {"filas": len(tr), "cuits": int(tr["cuit"].nunique()),
                           "periodos": [int(tr["periodo"].min()), int(tr["periodo"].max())],
                           "prevalencia": float(y_tr.mean())}}

    # ── entrenamiento ──
    if args.modelo in ("xgboost", "lightgbm"):
        sub = muestra_cuits(tr["cuit"].to_numpy(), args.frac_tuning, "tuning")
        print(f"\n[2] Búsqueda de hiperparámetros: {args.configs} configs × GroupKFold({args.folds}) "
              f"por CUIT sobre {sub.sum():,} filas...")
        tuning = tuning_boosting(args.modelo, tr.loc[sub, features].reset_index(drop=True),
                                 y_tr[sub], tr.loc[sub, "cuit"].to_numpy(),
                                 args.configs, args.folds)
        tuning.to_csv(out / "tuning.csv", index=False)
        mejor = tuning.iloc[0].to_dict()
        arboles = int(mejor.pop("arboles"))
        mejor.pop("pr_auc_cv"); mejor.pop("pr_auc_cv_sd")
        mejor = {k: (int(v) if k in ENTEROS else float(v)) for k, v in mejor.items()}
        print(f"  Mejor: {mejor}, {arboles} árboles")

        print(f"\n[3] Entrenando el modelo final con todo train...")
        params = params_boosting(args.modelo, mejor, peso_positivos(y_tr))
        if args.modelo == "xgboost":
            # QuantileDMatrix guarda los datos ya discretizados (~1 byte por valor)
            dtr = xgb.QuantileDMatrix(tr[features], y_tr, enable_categorical=True)
            del tr; gc.collect()
            modelo = xgb.train(params, dtr, arboles)
            del dtr; gc.collect()
            modelo.save_model(str(out / "modelo.json"))
            gan = modelo.get_score(importance_type="total_gain")
            spl = modelo.get_score(importance_type="weight")
            imp = pd.DataFrame({"variable": features,
                                "ganancia": [gan.get(f, 0.0) for f in features],
                                "splits": [spl.get(f, 0) for f in features]})
            predecir = lambda X: modelo.predict(xgb.DMatrix(X, enable_categorical=True))
        else:
            dtr = lgb.Dataset(tr[features], y_tr, free_raw_data=True)
            del tr; gc.collect()
            modelo = lgb.train(params, dtr, arboles)
            del dtr; gc.collect()
            modelo.save_model(str(out / "modelo.txt"))
            imp = pd.DataFrame({"variable": modelo.feature_name(),
                                "ganancia": modelo.feature_importance("gain"),
                                "splits": modelo.feature_importance("split")})
            predecir = lambda X: modelo.predict(X)
        imp["ganancia_pct"] = imp["ganancia"] / imp["ganancia"].sum() * 100
        imp.sort_values("ganancia", ascending=False).to_csv(out / "importancia.csv", index=False)
        resultado["hiperparametros"] = {**mejor, "arboles": arboles}
    else:
        print(f"\n[2] Entrenando regresión logística (Pipeline ajustado solo con train)...")
        modelo = pipeline_logistica(features)
        modelo.fit(tr[features], y_tr)
        del tr; gc.collect()
        joblib.dump(modelo, out / "modelo.joblib")
        nombres = modelo.named_steps["pre"].get_feature_names_out()
        coef = pd.DataFrame({"variable": nombres, "coeficiente": modelo.named_steps["lr"].coef_[0]})
        coef["abs"] = coef["coeficiente"].abs()
        coef.sort_values("abs", ascending=False).drop(columns="abs") \
            .to_csv(out / "importancia.csv", index=False)
        predecir = lambda X: modelo.predict_proba(X)[:, 1]

    # ── test (una sola vez) ──
    print(f"\n[4] Evaluando en test...")
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

    print(f"\n[5] Bootstrap por CUIT ({args.bootstrap} réplicas)...")
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
    resultado["minutos"] = round((time.time() - t0) / 60, 1)
    with open(out / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    # ── resumen ──
    r = resultado["test"]
    print(f"\n  Métricas por mes de scoring (media ± desvío entre {len(por_mes)} meses"
          f"{', IC 95% bootstrap por CUIT' if ic else ''}):")
    for c in cols_m:
        extra = f"  IC [{ic[c][0]:.4f}, {ic[c][1]:.4f}]" if c in ic else ""
        print(f"    {c:<14} {r['por_mes_media'][c]:.4f} ± {r['por_mes_sd'][c]:.4f}{extra}")
    for etiqueta, clave in (("Todas las filas juntas", "todas_las_filas"),
                            ("Una fila por CUIT", "una_fila_por_cuit")):
        mm = r[clave]
        print(f"  {etiqueta:<23}: AUC {mm['auc']:.4f}  PR-AUC {mm['pr_auc']:.4f}  "
              f"(prevalencia {mm['prevalencia']*100:.2f}%)")
    print(f"\n  Resultados en {out}  ({resultado['minutos']} min)")


if __name__ == "__main__":
    main()

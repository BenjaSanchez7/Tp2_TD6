# -*- coding: utf-8 -*-
from __future__ import annotations  # ← PRIMERA línea del archivo
from typing import List, Tuple, Optional, Dict

from dataclasses import dataclass
import os
import gc
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

# StratifiedGroupKFold (si está disponible) con fallback a GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGF = True
except Exception:
    HAS_SGF = False


@dataclass
class CFG:
    tree_method: str = "hist"
    learning_rate: float = 0.05
    max_depth: int = 7
    min_child_weight: float = 4.0
    subsample: float = 0.85
    colsample_bytree: float = 0.7
    reg_alpha: float = 0.5
    reg_lambda: float = 1.5
    gamma: float = 0.0
    n_splits: int = 5
    early_stopping_rounds: int = 200
    num_boost_round: int = 5000
    # ↓ Opcional: para ahorrar RAM quitamos columnas de texto crudo del keep
    DROP_TEXT_COLS: bool = True

pd.set_option("display.max_columns", None)

COMPETITION_PATH = ""

#?----------------------------------------------------
#? 1) CARGA DE DATA SETS
#?----------------------------------------------------
def loadData(data_dir: str):
    train = pd.read_csv(os.path.join(data_dir, "train_data.txt"), sep="\t", low_memory=False)
    test  = pd.read_csv(os.path.join(data_dir,  "test_data.txt"),  sep="\t", low_memory=False)
    return train, test
#Se cargan ambos datasets (train y test) para posteriormente poder trabajar con ellos


#?----------------------------------------------------
#? 2) CONCATENACION DE DATASETS Y TARGET
#?----------------------------------------------------
def unionAndTarget(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
#Conversion de timestamps para que las variables temporales se manejen como fechas y evitar problemas 
    for df in (train, test):
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        if "offline_timestamp" in df.columns:
            df["offline_timestamp"] = pd.to_datetime(df["offline_timestamp"], unit="s", utc=True, errors="coerce")

#Definicion de la variable target que vale 1 si la sesion termino porque el usuario presiono forward y 0 en cualquier otro caso
#Para el de test lo dejamos en nan ya que justamente lo queremos predecir
    train["target"] = (train["reason_end"] == "fwdbtn").astype(int) 
    test["target"] = np.nan

#flag para diferenciar si una fila pertenecer al train.txt(0) o al test.txt(1)
    train["is_test"] = 0
    test["is_test"]  = 1

    df = pd.concat([train, test], ignore_index=True)

#Se elimina la columna reason_end ya que nos dice porque termino la sesion lo cual esta 100% correlacionada con el target en el train.txt
    if "reason_end" in df.columns:
        df = df.drop(columns=["reason_end"])

    return df


# Helper TE OOF con smoothing
def oof_target_encode(df: pd.DataFrame, col: str, target_col: str,
                      groups: Optional[pd.Series] = None,
                      n_splits: int = 5, alpha: float = 50.0,
                      random_state: int = 42) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float32")

    trn_mask = (df["is_test"] == 0) & df[target_col].isin([0, 1])
    trn = df.loc[trn_mask, [col, target_col]]
    global_mean = float(trn[target_col].mean()) if len(trn) else 0.0

    enc_oof = pd.Series(np.nan, index=df.index, dtype="float32")

    # folds
    if groups is not None:
        gkf = GroupKFold(n_splits=n_splits)
        split_iter = gkf.split(trn, y=trn[target_col], groups=groups[trn_mask])
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = skf.split(trn, y=trn[target_col])

    for tr_idx, va_idx in split_iter:
        tr_fold = trn.iloc[tr_idx]
        va_fold_idx = trn.iloc[va_idx].index

        stats = tr_fold.groupby(col, observed=True)[target_col].agg(["sum", "count"])
        enc_map = (stats["sum"] + alpha * global_mean) / (stats["count"] + alpha)

        enc_oof.loc[va_fold_idx] = df.loc[va_fold_idx, col].map(enc_map).fillna(global_mean).astype("float32")

    # fit full train y asignar a TEST
    stats_all = trn.groupby(col, observed=True)[target_col].agg(["sum", "count"])
    enc_map_all = (stats_all["sum"] + alpha * global_mean) / (stats_all["count"] + alpha)
    test_idx = df.index[df["is_test"] == 1]
    enc_oof.loc[test_idx] = df.loc[test_idx, col].map(enc_map_all).fillna(global_mean).astype("float32")

    return enc_oof


#?----------------------------------------------------
#? 3) INGENIERIA DE ATRIBUTOS BASICA
#?----------------------------------------------------
def ingAtributos(df: pd.DataFrame) -> pd.DataFrame:
#Se ordenan los eventos por username y timestamp y se agrega un user_order para seguir el orden de interaccion de c/usuario con la app
    if {"username", "ts"}.issubset(df.columns):
        df = df.sort_values(["username", "ts"], kind="mergesort")
        df["user_order"] = df.groupby("username", observed=True).cumcount() + 1
    else:
        df["user_order"] = np.nan

# A partir de ts se guarda la hora del dia, el dia de la semana y una flag que vale 1 si es sabado o domingo y 0 en caso contrario para capturar
# patrones segun el dia y hora de la semana
    if "ts" in df.columns:
        df["hour"] = df["ts"].dt.hour.astype("Int64")
        df["dow"]  = df["ts"].dt.dayofweek.astype("Int64")
        df["is_weekend"] = (df["dow"] >= 5).astype(int)
    else:
        df["hour"] = df["dow"] = df["is_weekend"] = np.nan

# Para columnas booleanas, se transforman a enteros 0/1 (0 si no existen) para que sean comparables 
    for b in ["shuffle", "offline", "incognito_mode"]:
        if b in df.columns:
            df[b] = df[b].astype("Int64").fillna(0).astype(int)
        else:
            df[b] = 0

#Se cuenta cuantas veces aparece una cancion en el dataset y se guarda su "popularidad" en track_freq_all 
    is_train = (df.get("is_test", 1) == 0)

    if "spotify_track_uri" in df.columns:
        track_freq_tr = df.loc[is_train, "spotify_track_uri"].value_counts().astype("int32")
        df["track_freq_all"] = df["spotify_track_uri"].map(track_freq_tr).fillna(0).astype("int32")
    else:
        df["track_freq_all"] = 0

#Se cuenta cuantos eventos tiene c/usuario y se guarda su "actividad" en user_activity_all
    if "username" in df.columns:
        user_cnt_tr = df.loc[is_train, "username"].value_counts().astype("int32")
        df["user_activity_all"] = df["username"].map(user_cnt_tr).fillna(0).astype("int32")
    else:
        df["user_activity_all"] = 0

#Se calcula para c/usuario cuanto duro su evento previo y su evento siguiente (en caso de existir) (se completa con -1 en caso de no existir) 
#que sirve para ver la frecuencia de uso del usuario
    if {"username", "ts"}.issubset(df.columns):
        df["user_dt_prev"] = (
            df.groupby("username", observed=True)["ts"].diff().dt.total_seconds().fillna(-1).astype("float32")
        )
        df["user_dt_next"] = (
            df.groupby("username", observed=True)["ts"].diff(-1).abs().dt.total_seconds().fillna(-1).astype("float32")
        )
    else:
        df["user_dt_prev"] = df["user_dt_next"] = -1.0

    if "obs_id" in df.columns:
        df = df.sort_values("obs_id")

#Se calcula si el artista de la cancion es igual o distinto del artista del evento anterior
    if {"username", "master_metadata_album_artist_name"}.issubset(df.columns):
        prev_artist = df.groupby("username", observed=True)["master_metadata_album_artist_name"].shift(1)
        df["artist_change"] = (df["master_metadata_album_artist_name"] != prev_artist).astype(int)
        df["artist_change"] = df["artist_change"].fillna(0).astype(int)  
    else:
        df["artist_change"] = 0

#Calculamos el ratio historico de skips por usuario
#!Para evitar leakage se usan solo filas del train para calcular promedios
    # Media acumulada de skip por usuario (no-leaky): solo TRAIN y hasta el evento anterior
    if {"username", "target", "is_test"}.issubset(df.columns):
        mask_trn_row = (df["is_test"] == 0) & df["target"].isin([0, 1])
        global_skip = float(df.loc[mask_trn_row, "target"].mean()) if mask_trn_row.any() else 0.0

        # auxiliares solo para TRAIN
        df["target_trn"] = np.where(mask_trn_row, df["target"].astype("float32"), 0.0).astype("float32")
        df["is_trn"] = mask_trn_row.astype("int32")

        # acumulados previos por usuario (cumsum - valor_actual)
        prev_sum = (
            df.groupby("username", observed=True)["target_trn"].cumsum() - df["target_trn"]
        ).astype("float32")
        prev_cnt = (
            df.groupby("username", observed=True)["is_trn"].cumsum() - df["is_trn"]
        ).astype("float32")

        df["user_skip_cummean"] = np.where(prev_cnt > 0, prev_sum / prev_cnt, global_skip).astype("float32")

        # limpiar auxiliares
        df.drop(columns=["target_trn", "is_trn"], inplace=True)
    else:
        df["user_skip_cummean"] = 0.0

#Calculamos el ratio de skip de un track OOF
        # === Track skip ratio OOF ===
    if {"spotify_track_uri", "target", "is_test"}.issubset(df.columns):
        from sklearn.model_selection import KFold

        df["track_skip_ratio"] = np.nan

        mask_trn = (df["is_test"] == 0) & df["target"].isin([0, 1])
        idx_trn = np.where(mask_trn)[0]

        # Out-of-fold con KFold (no uses GroupKFold acá porque agrupa por user,
        # queremos promedios de track en general)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for tr_idx, va_idx in kf.split(idx_trn):
            tr_ids = idx_trn[tr_idx]
            va_ids = idx_trn[va_idx]

            means = (
                df.loc[tr_ids]
                  .groupby("spotify_track_uri")["target"]
                  .mean()
            )
            df.loc[va_ids, "track_skip_ratio"] = (
                df.loc[va_ids, "spotify_track_uri"].map(means)
            )

        # Rellenar test + valores no vistos con promedio global
        global_mean = float(df.loc[mask_trn, "target"].mean())
        df["track_skip_ratio"] = (
            df["track_skip_ratio"].fillna(global_mean).astype("float32")
        )
    else:
        df["track_skip_ratio"] = 0.0

# Target encoding OOF con smoothing para columnas de alta cardinalidad
    for te_col in ["platform", "master_metadata_album_artist_name"]:
        if te_col in df.columns and {"target","is_test"}.issubset(df.columns):
            groups_series = df["username"] if "username" in df.columns else None
            df[f"{te_col}_te"] = oof_target_encode(
                df, te_col, "target", groups=groups_series, n_splits=5, alpha=50.0, random_state=2025
            )

#Si hay un cambio entre una cancion y otra pero sucedieron en distintas sesiones no lo considero skip
    SESSION_GAP_SECONDS = 30 * 60  # 30 minutos

    #Detectar nueva sesión por usuario (si hay un gap grande de tiempo respecto al evento anterior)
    if {"username", "user_dt_prev"}.issubset(df.columns):
        df["new_session"] = (df["user_dt_prev"] > SESSION_GAP_SECONDS).fillna(False).astype(int)
    else:
        df["new_session"] = 0

    #Detectar cambio de tema respecto al evento anterior del mismo usuario
    if {"username", "spotify_track_uri"}.issubset(df.columns):
        prev_track = df.groupby("username", observed=True)["spotify_track_uri"].shift(1)
        df["track_changed"] = (df["spotify_track_uri"] != prev_track).fillna(False).astype(int)
    else:
        df["track_changed"] = 0

    #NO considerar "skip" si el cambio de tema ocurre entre sesiones
    df["not_skip_session_change"] = ((df["new_session"] == 1) & (df["track_changed"] == 1)).astype(int)

    # Identificador de sesión y posición/longitud dentro de la sesión
    if {"username", "new_session"}.issubset(df.columns):
        df["session_id"] = df.groupby("username", observed=True)["new_session"].cumsum().astype("int32")
        df["pos_in_session"] = df.groupby(["username", "session_id"], observed=True).cumcount().astype("int32")
        df["session_len"] = (
            df.groupby(["username", "session_id"], observed=True)["pos_in_session"].transform("max") + 1
        ).astype("int32")
    else:
        df["session_id"] = 0
        df["pos_in_session"] = 0
        df["session_len"] = 1

    # Historial inmediato del usuario (sin fuga)
    if {"username", "target"}.issubset(df.columns):
        # evento anterior del mismo user
        df["prev_target"] = (
            df.groupby("username", observed=True)["target"].shift(1).fillna(0).astype("int8")
        )

        # rolling de 5 eventos previos (shift para no ver el actual)
        tmp = (
            df.groupby("username", observed=True)["target"]
            .apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )
        # alinear índice (evita MultiIndex)
        tmp.index = df.index
        df["user_skip_roll5"] = tmp.fillna(0.0).astype("float32")
    else:
        df["prev_target"] = 0
        df["user_skip_roll5"] = 0.0

#Se arma una lista con las columnas utiles, el resto las ignoro y devuelvo una copia solo con las columnas que elegi dejar
    keep = [
        "obs_id", "is_test", "target",
        "username", "user_order", "user_activity_all", "user_dt_prev", "user_dt_next",
        "hour", "dow", "is_weekend",
        "spotify_track_uri", "master_metadata_track_name",
        "master_metadata_album_artist_name", "master_metadata_album_album_name",
        "episode_name", "episode_show_name", "spotify_episode_uri",
        "audiobook_title", "audiobook_uri", "audiobook_chapter_uri", "audiobook_chapter_title",
        "track_freq_all",
        "platform", "conn_country", "ip_addr",
        "shuffle", "offline", "incognito_mode", 
        #2 mas recientes ->
        "user_skip_cummean", "artist_change",
        "new_session", "track_changed", "not_skip_session_change", "track_skip_ratio",
        "session_id", "pos_in_session", "session_len",
        "prev_target", "user_skip_roll5"
    ]
    # Opcional: para ahorrar RAM, remover columnas de texto crudo (ya tenemos *_te si hicimos TE)
    if CFG.DROP_TEXT_COLS:
        drop_text = {
            "master_metadata_track_name",
            "master_metadata_album_artist_name",
            "master_metadata_album_album_name",
            "episode_name", "episode_show_name", "spotify_episode_uri",
            "audiobook_title", "audiobook_uri", "audiobook_chapter_uri", "audiobook_chapter_title",
            "spotify_track_uri",  # si no lo necesitás directo (tenés track_skip_ratio), lo podés sacar
        }
        keep = [c for c in keep if c not in drop_text]

    # quedate solo con las que existan
    keep = [c for c in keep if c in df.columns]

    # sumar dinámicamente los target encodings OOF si existen (p.ej. platform_te, master_metadata_album_artist_name_te)
    te_cols = [c for c in df.columns if c.endswith("_te")]
    keep = keep + te_cols

    # === Compactar dtypes ANTES de devolver para bajar memoria ===
    float_cols = df.select_dtypes(include=["float64"]).columns.tolist()
    if float_cols:
        df[float_cols] = df[float_cols].astype("float32")

    int64_cols = df.select_dtypes(include=["int64"]).columns.tolist()
    preserve_int64 = {"obs_id"}  # mantener IDs largos si querés
    to_int32 = [c for c in int64_cols if c not in preserve_int64]
    if to_int32:
        df[to_int32] = df[to_int32].astype("int32")

    # Devolver SIN copia profunda para evitar consolidación costosa de bloques
    return df.loc[:, keep].reset_index(drop=True)


#?----------------------------------------------------
#? 4) ENCODING VARIABLES CATEGORICAS
#?----------------------------------------------------
def encodeCategoricas(df: pd.DataFrame):
    is_test = df["is_test"].to_numpy()
    y = df["target"].to_numpy()

    X = df.drop(columns=["is_test", "target", "obs_id", "username"], errors="ignore")

    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype) == "string"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    encoder = None
    if len(cat_cols) > 0:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        trn_mask = (df["is_test"] == 0)

        # fit SOLO con train
        encoder.fit(X.loc[trn_mask, cat_cols])

        # transform train y test por separado para evitar reindex raros
        X_cat_tr = encoder.transform(X.loc[trn_mask, cat_cols])
        X_cat_te = encoder.transform(X.loc[~trn_mask, cat_cols])

        X_cat_tr = pd.DataFrame(X_cat_tr, columns=cat_cols, index=X.loc[trn_mask].index)
        X_cat_te = pd.DataFrame(X_cat_te, columns=cat_cols, index=X.loc[~trn_mask].index)

        X_cat = pd.concat([X_cat_tr, X_cat_te], axis=0).sort_index()
        X = pd.concat([X[num_cols], X_cat], axis=1)

    feature_names = X.columns.tolist()
    return X, y, is_test, feature_names, encoder


#?----------------------------------------------------
#? 5) TRAINING Y VALIDATION
#?----------------------------------------------------
def trainAndValidation(
    df: pd.DataFrame,
    feature_names: List[str],
    groups: pd.Series,  # se ignora; se reconstruye adentro a partir de df
    n_splits: int = 5,
    params: Optional[Dict] = None,
    seeds: List[int] = (42,),
) -> Tuple[np.ndarray, np.ndarray]:
    
    # máscaras
    trn_idx = df["is_test"].values == 0
    tst_idx = ~trn_idx

    # datos (solo TRAIN para X/y)
    X = df.loc[trn_idx, feature_names].to_numpy()
    y = df.loc[trn_idx, "target"].to_numpy().astype(np.int32)
    X_test = df.loc[tst_idx, feature_names].to_numpy()

    # === construir groups SOLO para TRAIN (alineado 1–a–1 con X, y), factorizar ===
    if "username" in df.columns:
        groups_trn = pd.factorize(
            df.loc[trn_idx, "username"].astype(str).fillna("unknown")
        )[0]
    else:
        groups_trn = pd.factorize(
            df.loc[trn_idx, "obs_id"].astype(str).fillna("unknown")
        )[0]

    # sanity check
    assert len(groups_trn) == len(y), f"len(groups_trn)={len(groups_trn)} != len(y)={len(y)}"

    # desbalance
    pos = y.sum()
    neg = len(y) - pos
    scale_pos_weight = float(neg / max(pos, 1))

    # params base
    base_params = dict(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method=CFG.tree_method,
        learning_rate=CFG.learning_rate,
        max_depth=CFG.max_depth,
        min_child_weight=CFG.min_child_weight,
        subsample=CFG.subsample,
        colsample_bytree=CFG.colsample_bytree,
        reg_alpha=CFG.reg_alpha,
        reg_lambda=CFG.reg_lambda,
        gamma=CFG.gamma,
        scale_pos_weight=scale_pos_weight,
        verbosity=0,
    )
    if params:
        base_params.update(params)

    oof = np.zeros(len(y), dtype=np.float32)
    test_pred_bag = np.zeros((len(X_test), len(seeds)), dtype=np.float32)

    for s_idx, seed in enumerate(seeds):
        base_params["seed"] = seed

        # splitter (StratifiedGroupKFold si está; si no, GroupKFold)
        if HAS_SGF:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            split_iter = splitter.split(X, y, groups=groups_trn)
        else:
            splitter = GroupKFold(n_splits=n_splits)
            split_iter = splitter.split(X, y, groups=groups_trn)

        fold_preds = np.zeros(len(X_test), dtype=np.float32)

        for fold, (tr_idx, va_idx) in enumerate(split_iter):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
            dva = xgb.DMatrix(X_va, label=y_va, feature_names=feature_names)
            dte = xgb.DMatrix(X_test, feature_names=feature_names)

            bst = xgb.train(
                params=base_params,
                dtrain=dtr,
                num_boost_round=CFG.num_boost_round,
                evals=[(dtr, "tr"), (dva, "va")],
                early_stopping_rounds=CFG.early_stopping_rounds,
                verbose_eval=False,
            )

            va_pred = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
            oof[va_idx] = (oof[va_idx] + va_pred) if s_idx > 0 else va_pred

            fold_pred = bst.predict(dte, iteration_range=(0, bst.best_iteration + 1))
            fold_preds += fold_pred

            auc = roc_auc_score(y_va, va_pred)
            print(f"[seed {seed}] Fold {fold+1}/{n_splits} AUC = {auc:.5f} | best_iter = {bst.best_iteration}")

            del dtr, dva
            gc.collect()

        fold_preds /= n_splits
        test_pred_bag[:, s_idx] = fold_preds

    test_pred = test_pred_bag.mean(axis=1)
    if len(seeds) > 1:
        oof /= len(seeds)

    overall_auc = roc_auc_score(y, oof)
    print(f"OOF AUC: {overall_auc:.5f}")

    return oof, test_pred


#?----------------------------------------------------
#? 6) PIPELINE
#?----------------------------------------------------
def main():
    print("=== TP2 XGBoost baseline ===")
    train, test = loadData(COMPETITION_PATH)
    df = unionAndTarget(train, test)
    df = ingAtributos(df)

    # 1) encode
    X, y, is_test, feature_names, _ = encodeCategoricas(df)

    # 2) reconstruir un df "encodeado" con las columnas que usa trainAndValidation
    X_df = pd.DataFrame(X, columns=feature_names, index=df.index)
    df_enc = pd.concat(
        [df[["obs_id", "is_test", "target", "username"]].reset_index(drop=True),
         X_df.reset_index(drop=True)],
        axis=1
    )

    # 3) llamar a trainAndValidation con df_enc (no pases X, y, is_test)
    oof, test_pred = trainAndValidation(
        df=df_enc,
        feature_names=feature_names,
        groups=df_enc["username"],     # se reconstruye adentro con seguridad
        n_splits=CFG.n_splits,
        params=None,
        seeds=[42],
    )

    # AUC OOF antes del submit
    mask_trn = (df_enc["is_test"] == 0)
    auc_oof = roc_auc_score(df_enc.loc[mask_trn, "target"].astype(int), oof)
    print(f"AUC OOF: {auc_oof:.5f}")

    # Submit
    obs_id_test = df_enc.loc[df_enc["is_test"] == 1, "obs_id"].to_numpy()
    sub = pd.DataFrame({"obs_id": obs_id_test, "pred_proba": test_pred})
    sub.to_csv("submission_xgb2.csv", index=False)
    print("→ submission_xgb2.csv generado OK")
    print("=== Fin ===")


if __name__ == "__main__":
    main()

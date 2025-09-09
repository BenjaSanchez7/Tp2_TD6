import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

pd.set_option("display.max_columns", None)

# ==== Config ====
COMPETITION_PATH = "."               # ajustar si es necesario
MODEL_KIND = "xgb"                   # "xgb" o "rf"
VAL_FRACTION_PER_USER = 0.20         # último 20% de cada usuario → validación temporal
USER_HOLDOUT_FRAC = 0.20             # % de usuarios reservados para validación por usuario
RANDOM_STATE = 42


def load_competition_datasets(data_dir, sample_frac=None, random_state=None):
    print("Loading competition datasets from:", data_dir)
    train_file = os.path.join(data_dir, "train_data.txt")
    test_file  = os.path.join(data_dir, "test_data.txt")

    train_df = pd.read_csv(train_file, sep="\t", low_memory=False)
    if sample_frac is not None:
        train_df = train_df.sample(frac=sample_frac, random_state=random_state)

    test_df  = pd.read_csv(test_file,  sep="\t", low_memory=False)

    combined = pd.concat([train_df, test_df], ignore_index=True)
    print(f"  → Concatenated DataFrame: {combined.shape[0]} rows")
    return combined

def build_artist_stats(df, mask_rows, key_col, target):
    """
    Calcula cnt y fwdr del artista usando SOLO las filas donde mask_rows es True (tu TR).
    """
    aux = pd.DataFrame({
        key_col: df[key_col].astype("string"),
        "target": target.astype("float32")
    })
    aux_tr = aux[mask_rows].dropna(subset=["target"])
    grp = aux_tr.groupby(key_col, observed=True)["target"]
    stats = pd.DataFrame({
        "cnt": grp.size(),
        "fwdr": grp.mean()
    }).reset_index()
    prior = float(aux_tr["target"].mean()) if len(aux_tr) else 0.0
    return stats, prior

def attach_stats(X_base, stats, prior):
    """
    Une stats al DataFrame de features y completa con prior/0 para categorías no vistas.
    """
    X = X_base.merge(stats, on="master_metadata_album_artist_name", how="left")
    X["cnt"]  = X["cnt"].fillna(0).astype("int32")
    X["fwdr"] = X["fwdr"].fillna(prior).astype("float32")
    return X


def cast_column_types(df):
    print("Casting column types and parsing datetime fields...")
    dtype_map = {
        "platform": "category",
        "conn_country": "category",
        "ip_addr": "category",
        "master_metadata_track_name": "category",
        "master_metadata_album_artist_name": "category",
        "master_metadata_album_album_name": "category",
        "reason_end": "category",
        "username": "category",
        "spotify_track_uri": "string",
        "episode_name": "string",
        "episode_show_name": "string",
        "spotify_episode_uri": "string",
        "audiobook_title": "string",
        "audiobook_uri": "string",
        "audiobook_chapter_uri": "string",
        "audiobook_chapter_title": "string",
        "shuffle": "boolean",
        "offline": "boolean",
        "incognito_mode": "boolean",
        "obs_id": "int64",
    }
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["offline_timestamp"] = pd.to_datetime(df["offline_timestamp"], unit="s", errors="coerce", utc=True)
    df = df.astype(dtype_map)
    print("  → Column types cast successfully.")
    return df


def make_target_and_mask(df):
    """
    Crea target SOLO en train y genera is_test sin fugar.
    En test no hay reason_end, así que target queda NA.
    """
    print("Creating 'target' and 'is_test' columns (no leakage)...")
    is_test = df["reason_end"].isna()
    target = (df["reason_end"] == "fwdbtn").astype("Int8")  # 1/0 en train
    target[is_test] = pd.NA  # no inventar etiquetas en test
    # NO borramos acá columnas que necesitamos para features (lo haremos después si hace falta)
    return is_test.to_numpy(), target


def add_feature_blocks(df, is_test_mask, target):
    """
    Agrega features sin fuga (SIN cnt/fwdr aquí):
      - user_order (posición secuencial por usuario)
      - hour, dow (temporal)
      - dt_prev (segundos desde la reproducción anterior del mismo usuario, cap 48h)
    """
    # Orden temporal por usuario y posición en la sesión
    df = df.sort_values(["username", "ts"])
    df["user_order"] = df.groupby("username", observed=True).cumcount() + 1

    # Variables temporales
    df["hour"] = df["ts"].dt.hour.astype("int16")
    df["dow"]  = df["ts"].dt.dayofweek.astype("int8")  # 0=lunes

    # Delta de tiempo con el anterior del mismo usuario
    dt = df["ts"] - df.groupby("username", observed=True)["ts"].shift(1)
    df["dt_prev"] = dt.dt.total_seconds()
    median_dt = df.loc[~is_test_mask, "dt_prev"].median(skipna=True)
    if pd.isna(median_dt):
        median_dt = 0.0
    df["dt_prev"] = df["dt_prev"].fillna(median_dt).clip(0, 48 * 3600).astype("int32")

    # Volver a ordenar por obs_id
    df = df.sort_values(["obs_id"]).reset_index(drop=True)

    feats = [
        "obs_id",                   # para submit
        "user_order", "hour", "dow",
        "dt_prev",
        "platform", "conn_country",
        "master_metadata_album_artist_name",
        "username",
    ]
    return df[feats].copy(), df["username"].astype("string").to_numpy()


def encode_categoricals(X_full, train_mask):
    """
    Codifica categóricas con categorías aprendidas SOLO del train.
    Categorías no vistas en val/test → -1 (evita fuga).
    """
    X = X_full.copy()
    for col in X.columns:
        if X[col].dtype == "object" or str(X[col].dtype) in ("string",):
            X[col] = X[col].astype("category")

    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
    for col in cat_cols:
        train_cats = X.loc[~train_mask, col].cat.categories
        X[col] = X[col].cat.set_categories(train_cats).cat.codes.astype("int32")
    return X


def train_classifier(X_train, y_train, kind="xgb", params=None):
    """
    Entrena RF o XGB según 'kind'.
    """
    if kind == "xgb":
        if not _HAS_XGB:
            raise RuntimeError("xgboost no está instalado. Cambiá MODEL_KIND='rf' o pip install xgboost")
        print("Training XGBoost...")
        default_params = {
            "n_estimators": 700,
            "max_depth": 6,
            "learning_rate": 0.06,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 4,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "random_state": RANDOM_STATE,
            "eval_metric": "auc",
            "n_jobs": -1,
            "tree_method": "hist",
        }
        if params: default_params.update(params)
        model = XGBClassifier(**default_params)
    else:
        print("Training RandomForest...")
        default_params = {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
            "bootstrap": True,
        }
        if params: default_params.update(params)
        model = RandomForestClassifier(**default_params)

    model.fit(X_train, y_train)
    print("  → Model training complete.")
    return model


def main():
    print("=== Starting improved pipeline (no-leak stats + double validation) ===")
    # 1) Carga y casteo
    df = load_competition_datasets(COMPETITION_PATH, sample_frac=None, random_state=RANDOM_STATE)
    df = cast_column_types(df)

    # 2) Target/Mask sin fuga (target solo en train)
    is_test_mask, target = make_target_and_mask(df)
    train_mask = ~is_test_mask

    # 3) Features seguras (sin cnt/fwdr acá)
    X_full, username = add_feature_blocks(df, is_test_mask, target)  # trae: obs_id, user_order, hour, dow, dt_prev, platform, conn_country, artist, username

    # 4) Validación temporal: último 20% por usuario (dentro de TRAIN)
    user_order = X_full["user_order"].to_numpy()
    val_mask_temporal = np.zeros(len(df), dtype=bool)
    aux = pd.DataFrame({
        "idx": np.arange(len(df)),
        "username": username,
        "user_order": user_order,
        "is_train": train_mask
    })
    aux_train = aux[aux["is_train"]]
    cut_by_user = aux_train.groupby("username", observed=True)["user_order"].quantile(1.0 - VAL_FRACTION_PER_USER)
    aux_train = aux_train.join(cut_by_user, on="username", rsuffix="_cut")
    val_mask_train = aux_train["user_order"] >= aux_train["user_order_cut"]
    val_indices = aux_train.loc[val_mask_train, "idx"].to_numpy()
    val_mask_temporal[val_indices] = True

    tr_mask_temp = train_mask & (~val_mask_temporal)
    va_mask_temp = train_mask & ( val_mask_temporal)

    # 5) Validación user-holdout: 20% de usuarios nunca vistos en train
    rng = np.random.RandomState(RANDOM_STATE)
    users_train = pd.Series(username[train_mask]).unique()
    holdout_users = pd.Series(users_train).sample(frac=USER_HOLDOUT_FRAC, random_state=RANDOM_STATE).astype(str).tolist()

    user_holdout_mask = np.zeros(len(df), dtype=bool)
    user_holdout_mask[train_mask] = pd.Series(username[train_mask]).isin(holdout_users).to_numpy()

    tr_mask_user = train_mask & (~user_holdout_mask)
    va_mask_user = train_mask & ( user_holdout_mask)

    # 6) ===== Recalcular stats por ARTISTA sin fuga para cada split =====
    # (A) Temporal: stats con SOLO tr_mask_temp
    artist_stats_temp, prior_temp = build_artist_stats(
        df, tr_mask_temp, "master_metadata_album_artist_name", target
    )
    X_full_temp = attach_stats(X_full, artist_stats_temp, prior_temp)   # ahora X_full_temp SÍ tiene cnt/fwdr (construidos solo con TR temporal)

    # Codificación categórica consistente (categorías desde TODO el train, sin mirar test)
    X_full_temp_enc = encode_categoricals(X_full_temp, train_mask=train_mask)

    # Separar matrices TEMPORALES
    obs_id_temp = X_full_temp_enc["obs_id"].to_numpy()
    X_enc_noid_temp = X_full_temp_enc.drop(columns=["obs_id"])
    X_tr  = X_enc_noid_temp[tr_mask_temp];  y_tr  = target[tr_mask_temp].astype("int8")
    X_val = X_enc_noid_temp[va_mask_temp];  y_val = target[va_mask_temp].astype("int8")

    # (B) User-holdout: stats con SOLO tr_mask_user
    artist_stats_user, prior_user = build_artist_stats(
        df, tr_mask_user, "master_metadata_album_artist_name", target
    )
    X_full_user = attach_stats(X_full, artist_stats_user, prior_user)

    X_full_user_enc = encode_categoricals(X_full_user, train_mask=train_mask)

    # Separar matrices USER-HOLDOUT
    X_enc_noid_user = X_full_user_enc.drop(columns=["obs_id"])
    X_tr_u  = X_enc_noid_user[tr_mask_user];  y_tr_u  = target[tr_mask_user].astype("int8")
    X_val_u = X_enc_noid_user[va_mask_user];  y_val_u = target[va_mask_user].astype("int8")

    # (C) TEST: usamos las stats del TR temporal (con lo que entrenamos el modelo final)
    X_test_temp   = X_enc_noid_temp[is_test_mask]
    test_obs_ids  = obs_id_temp[is_test_mask]

    print(f"  → Train (temporal) rows: {X_tr.shape[0]}  |  Val (temporal) rows: {X_val.shape[0]}  |  Test rows: {X_test_temp.shape[0]}")
    print(f"  → Train (user-holdout) rows: {X_tr_u.shape[0]}  |  Val (user-holdout) rows: {X_val_u.shape[0]}")

    # 7) Entrenar con el split temporal (más datos que el user-holdout)
    model = train_classifier(X_tr, y_tr, kind=MODEL_KIND)

    # 8) AUCs de validación (ya sin fuga)
    print("Evaluating ROC-AUC on validation sets (no-leak stats)...")
    val_auc_temp = roc_auc_score(y_val,   model.predict_proba(X_val)[:, 1])
    val_auc_user = roc_auc_score(y_val_u, model.predict_proba(X_val_u)[:, 1])
    print(f"ROC-AUC (validación temporal por usuario): {val_auc_temp:.5f}")
    print(f"ROC-AUC (validación por usuario holdout): {val_auc_user:.5f}")

    # 9) Importancias
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            feature_names = X_tr.columns
            imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            print("\nTop 25 feature importances:")
            print(imp_series.head(25))
    except Exception:
        pass

    # 10) Predicción y export (con features temporales y stats del TR temporal)
    print("Generating predictions for test set...")
    preds_proba = model.predict_proba(X_test_temp)[:, 1]
    preds_df = pd.DataFrame({"obs_id": test_obs_ids, "pred_proba": preds_proba})
    preds_df.to_csv("modelo_benchmark.csv", index=False)
    print("  → Predictions written to 'modelo_benchmark.csv'")
    print("=== Pipeline complete ===")



if __name__ == "__main__":
    main()

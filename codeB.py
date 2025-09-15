import os
import gc
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder


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
    if {"username", "target", "is_test"}.issubset(df.columns):
        mask_trn = (df["is_test"] == 0) & df["target"].isin([0, 1])
        user_skip_ratio = df.loc[mask_trn].groupby("username", observed=True)["target"].mean().astype("float32")
        df["user_skip_ratio"] = df["username"].map(user_skip_ratio).astype("float32")
        global_skip = float(df.loc[mask_trn, "target"].mean()) if mask_trn.any() else 0.0
        df["user_skip_ratio"] = df["user_skip_ratio"].fillna(global_skip).astype("float32")
    else:
        df["user_skip_ratio"] = 0.0


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
        "user_skip_ratio", "artist_change"
        "new_session", "track_changed", "not_skip_session_change", "track_skip_ratio"
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()

#?----------------------------------------------------
#? 4) ENCODING VARIABLES CATEGORICAS
#?----------------------------------------------------
def encodeCategoricas(df: pd.DataFrame):
#Se guarda la flag que indica si una fila pertenece a train(0) o a test(1)   
#Se guarda el target 
    is_test = df["is_test"].to_numpy()
    y = df["target"].to_numpy()

#Se eliminan todas las columnas que no sean features para luego poder armar la matriz de variables explicativas
#Se separan todas las columnas restantes en categoricas (cat_cols) y numericas (num_cols)
    X = df.drop(columns=["is_test", "target", "obs_id"], errors="ignore")

    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype) == "string"]
    num_cols = [c for c in X.columns if c not in cat_cols]

#Se usa OrdinalEncoder que para cada columna categorica la mapea a un numero entero consistente con el resto del dataset
#los valores missing o nulos quedan en -1. De esta forma queda todo en numeros para que los arboles de XGBoost funcionen correctamente
#Finalmente se concatenan nuevamente las num_cols y las ahora X_cat (categoricas codificadas)
    encoder = None
    if len(cat_cols) > 0:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        X_cat = encoder.fit_transform(X[cat_cols])
        X_cat = pd.DataFrame(X_cat, columns=cat_cols, index=X.index)
        X = pd.concat([X[num_cols], X_cat], axis=1)

    feature_names = X.columns.tolist()
    return X, y, is_test, feature_names, encoder
#X = matriz, y = vector target, is_test, featurne_names = nombre de cada columna, encoder


#?----------------------------------------------------
#? 5) TRAINING Y VALIDATION
#?----------------------------------------------------
def trainAndValidation(
    X: pd.DataFrame,
    y: np.ndarray,
    is_test: np.ndarray,
    feature_names,
    groups: np.ndarray,
    params: dict = None,
    n_splits: int = 5,
):
#Cra una mascara booleana is_test y separa: solo filas de train, solo filas de test y targets alineados a filas de train
    is_test_bool = pd.Series(is_test, index=X.index).astype(bool)

    X_trn = X.loc[~is_test_bool]
    X_tst = X.loc[is_test_bool]
    y_trn = y[~is_test_bool.values]

    pos_ratio = float(np.mean(y_trn))
    auto_spw = ((1.0 - pos_ratio)/ max(pos_ratio, 1e-8)) if 0.0 <pos_ratio <1.0 else 1.0
#Se definen hiperparametros base para XGBoost
    base_params = dict(
        max_depth=7,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.7,
        min_child_weight=1,
        reg_alpha=0.0,
        reg_lambda=1.5,
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
    )
    if params:
        base_params.update(params)

#Calculo el porcentaje de positivos en mi train y define scale_pos_weight para ver el desbalance del dataset
    if "scale_pos_weight" not in base_params:
        base_params["scale_pos_weight"] = auto_spw
    print(f"[Info] pos_ratio={pos_ratio:.4f}  -> scale_pos_weight={base_params['scale_pos_weight']:.3f}")


#Configura el GroupKFold y usa n_splits para no mezclar eventos del mismo usuario entre train y validacion
#itera por folds y en cada uno crea X_tr, X_va, y_tr, y_va
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(X_trn), dtype=np.float32)
    test_pred = np.zeros(len(X_tst), dtype=np.float32)
    fold_scores = []
    models = []

    groups_trn = groups[~is_test_bool.values]
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_trn, y_trn, groups=groups_trn), 1):
        X_tr, y_tr = X_trn.iloc[tr_idx], y_trn[tr_idx]
        X_va, y_va = X_trn.iloc[va_idx], y_trn[va_idx]

#Entrena XGBoost con early stopping
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dvalid = xgb.DMatrix(X_va, label=y_va)
        dtest  = xgb.DMatrix(X_tst)

        watchlist = [(dtrain, "train"), (dvalid, "valid")]
        model = xgb.train(
            params=base_params,
            dtrain=dtrain,
            num_boost_round=5000,
            evals=watchlist,
            early_stopping_rounds=150,
            verbose_eval=False,
        )

#Determina cuantos arboles usar (en caso de no haber early stopping usa todos)
        best_it = model.best_iteration
        use_range = None if best_it is None else (0, best_it+1)

#Prediccion de OOF (out of fold) usando hasta best_iteration
        if use_range is None:
            p_va = model.predict(dvalid)
        else:
            p_va = model.predict(dvalid, iteration_range = use_range)
    
        oof[va_idx] = p_va
        auc = roc_auc_score(y_va, p_va)
        fold_scores.append(auc)
        models.append(model)

        print(f"[Fold {fold}] AUC = {auc:.5f}  best_iter={model.best_iteration}")
    
#Prediccion de TEST usando solo hasta best_iteration

        if use_range is None:
            test_pred += model.predict(dtest) / n_splits
        else:
            test_pred += model.predict(dtest, iteration_range = use_range) / n_splits

        del X_tr, y_tr, X_va, y_va, p_va
        gc.collect()

#Printea metricas importantes
    print(f"CV AUC (GroupKFold username): {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")

    importances = np.zeros(len(feature_names), dtype=np.float64)
    for m in models:
        gain = m.get_score(importance_type="gain")
        for i, f in enumerate(feature_names):
            importances[i] += gain.get(f, 0.0)
    importances /= max(1, len(models))
    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    print("\nTop 20 features por 'gain':")
    print(imp_series.head(20))

    return test_pred, imp_series


#?----------------------------------------------------
#? 6) PIPELINE
#?----------------------------------------------------
def main():
    print("=== TP2 XGBoost baseline ===")
    train, test = loadData(COMPETITION_PATH)
    df = unionAndTarget(train, test)
    df = ingAtributos(df)

    if "username" in df.columns:
        groups = df["username"].astype(str).fillna("unknown").to_numpy()
    else:
        groups = df["obs_id"].astype(str).to_numpy()

    X, y, is_test, feature_names, _ = encodeCategoricas(df)

    test_pred, importances = trainAndValidation(
        X=X, y=y, is_test=is_test,
        feature_names=feature_names, groups=groups,
        params=None,
        n_splits=5
    )

    obs_id_test = df.loc[df["is_test"] == 1, "obs_id"].to_numpy()
    sub = pd.DataFrame({"obs_id": obs_id_test, "pred_proba": test_pred})
    sub.to_csv("submission_xgb.csv", index=False)
    print("→ submission_xgb.csv generado OK")
    print("=== Fin ===")


if __name__ == "__main__":
    main()

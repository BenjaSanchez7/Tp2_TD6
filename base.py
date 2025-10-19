import os
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)

# Adjust this path if needed
COMPETITION_PATH =""


def load_competition_datasets(data_dir, sample_frac=None, random_state=None):
    """
    Load train and test datasets, optionally sample a fraction of the training set,
    concatenate, and reset index.
    """
    print("Loading competition datasets from:", data_dir)
    train_file = os.path.join(data_dir, "train_data.txt")
    test_file = os.path.join(data_dir, "test_data.txt")

    # Load training data and optionally subsample
    train_df = pd.read_csv(train_file, sep="\t", low_memory=False)
    if sample_frac is not None:
        train_df = train_df.sample(frac=sample_frac, random_state=random_state)

    # Load test data
    test_df = pd.read_csv(test_file, sep="\t", low_memory=False)

    # Concatenate and reset index
    combined = pd.concat([train_df, test_df], ignore_index=True)
    print(f"  → Concatenated DataFrame: {combined.shape[0]} rows")
    return combined


def cast_column_types(df):
    """
    Cast columns to efficient dtypes and parse datetime fields.
    """
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
        "shuffle": bool,
        "offline": bool,
        "incognito_mode": bool,
        "obs_id": int,
    }

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["offline_timestamp"] = pd.to_datetime(
        df["offline_timestamp"], unit="s", errors="coerce", utc=True
    )
    df = df.astype(dtype_map)
    print("  → Column types cast successfully.")
    return df


def split_train_test(X, y, test_mask):
    """
    Split features and labels into train/test based on mask.
    """
    print("Splitting data into train/test sets...")

    # Asegurar máscara booleana alineada con X
    mask = pd.Series(test_mask, index=X.index).astype(bool)

    # No usar estas columnas como features (por seguridad)
    X = X.drop(columns=["is_test", "obs_id"], errors="ignore")

    # Alinear y con X e ir a numpy para el modelo
    y_series = y if isinstance(y, pd.Series) else pd.Series(y, index=X.index)

    # Particiones
    X_train = X.loc[~mask]
    X_test  = X.loc[mask]
    y_train = y_series.loc[~mask].to_numpy()
    y_test  = None  # en test no hay labels

    print(f"  → Training set: {X_train.shape[0]} rows")
    print(f"  → Test set:     {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test


def train_classifier(X_train, y_train, params=None):
    """
    Train a Classifier 
    """
    print("Training model...")
    
    default_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": 42,
        "bootstrap": True,
    }
    rf_params = default_params.copy()
    if params:
        rf_params.update(params)

    model = RandomForestClassifier(**rf_params)

    print("  → Fitting RandomForestClassifier...")
    model.fit(X_train, y_train)
    print("  → Model training complete.")
    return model





def main():
    print("=== Starting pipeline ===")

    # Load and preprocess data
    df = load_competition_datasets(
        COMPETITION_PATH, sample_frac=0.2, random_state=1234
    )
    df = cast_column_types(df)

    # Generate user order column
    df = df.sort_values(["username", "ts"])
    df["user_order"] = df.groupby("username", observed=True).cumcount() + 1
    df = df.sort_values(["obs_id"])

    
    # Create target and test mask

    print("Creating 'target' and 'is_test' columns...")
    df["target"] = (df["reason_end"] == "fwdbtn").astype(int)
    df["is_test"] = df["reason_end"].isna()
    df.drop(columns=["reason_end"], inplace=True)
    print("  → 'target' and 'is_test' created, dropped 'reason_end' column.")

    to_keep = [
        "obs_id",
        "target",
        "is_test",
        "user_order",
        #Agrege esto
        "shuffle",
        "offline",
        "incognito_mode"
    ] #MODIFICAR
    df = df[to_keep]
    obs_id_test = df.loc[df["is_test"], "obs_id"].to_numpy()

   # Build feature matrix and get feature names
    y = df["target"].to_numpy()
    X = df.drop(columns=["target", "is_test", "obs_id"])
    feature_names = X.columns
    test_mask = df["is_test"].to_numpy()

    # Split data
    X_train, X_test, y_train, _ = split_train_test(X, y, test_mask)

    # Train model
    model = train_classifier(X_train, y_train)

    # Display top 20 feature importances
    print("Extracting and sorting feature importances...")
    importances = model.feature_importances_
    imp_series = pd.Series(importances, index=feature_names)
    imp_sorted = imp_series.sort_values(ascending=False)
    print("\nTop 20 feature importances:")
    print(imp_sorted.head(20))

    # Predict on test set
    print("Generating predictions for test set...")
    test_obs_ids = obs_id_test
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds_df = pd.DataFrame({"obs_id": test_obs_ids, "pred_proba": preds_proba})
    preds_df.to_csv("modelo_benchmark.csv", index=False, sep=",")
    print(f"  → Predictions written to 'modelo_benchmark.csv'")

    print("=== Pipeline complete ===")


if __name__ == "__main__":
    main()

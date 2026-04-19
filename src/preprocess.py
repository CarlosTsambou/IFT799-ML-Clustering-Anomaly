# src/preprocess.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def load_hiseq():
    """Charge les données Hi-Seq et renvoie (X, y) en numpy."""
    data_path = os.path.join(DATA_DIR, "hiseq_data.csv")
    labels_path = os.path.join(DATA_DIR, "hiseq_labels.csv")

    df_X = pd.read_csv(data_path)
    df_y = pd.read_csv(labels_path)

    # 1) X : on enlève la première colonne (les IDs 'sample_0', etc.)
    #    Toutes les colonnes après sont les gènes (numériques)
    df_X = df_X.drop(columns=[df_X.columns[0]])
    X = df_X.values.astype(float)

    # 2) y : on prend la colonne 'Class' (les types de cancer)
    #    en ignorant la colonne d'ID 'Unnamed: 0'
    if "Class" in df_y.columns:
        y = df_y["Class"].values
    else:
        # fallback si jamais le nom de colonne a changé
        y = df_y.iloc[:, -1].values

    return X, y


def normalize_features(X):
    """Standardise les features (moyenne 0, variance 1)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def reduce_pca(X, n_components=100):
    """Réduction de dimension via ACP (PCA) à 100 dims."""
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

def reduce_umap(X, n_components=100, n_neighbors=15, min_dist=0.1):
    """Réduction de dimension via UMAP à 100 dims."""
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42
    )
    X_umap = reducer.fit_transform(X)
    return X_umap, reducer

def prepare_hiseq_all_spaces():
    """
    Prépare les 3 versions des données Hi-Seq :
    - X_full : données normalisées complètes
    - X_pca : 100 dims via ACP
    - X_umap : 100 dims via UMAP
    """
    X, y = load_hiseq()
    X_scaled, scaler = normalize_features(X)
    X_pca, pca = reduce_pca(X_scaled, n_components=100)
    X_umap, umap_model = reduce_umap(X_scaled, n_components=100)

    feature_spaces = {
        "full": X_scaled,
        "pca100": X_pca,
        "umap100": X_umap,
    }
    models = {
        "scaler": scaler,
        "pca": pca,
        "umap": umap_model,
    }
    return feature_spaces, y, models


def load_ecg():
    """Charge le fichier ecg.npz et renvoie (X, y)."""
    ecg_path = os.path.join(DATA_DIR, "ecg.npz")
    data = np.load(ecg_path)
    df_ecg = data['ecg']      # shape (4998, 141) normalement
    X = df_ecg[:, :-1]        # 140 features
    y = df_ecg[:, -1]         # dernière colonne = label (0 = normal, 1 = anomalie)
    return X.astype(float), y.astype(int)

def split_ecg_train_val_test(random_state=42):
    """
    Split selon les proportions demandées :
    - train : 60% des normaux
    - test : 30% des normaux + 80% des anormaux
    - val  : 10% des normaux + 20% des anormaux
    """
    X, y = load_ecg()
    X_norm = X[y == 0]
    X_anom = X[y == 1]

    # On shuffle via train_test_split en deux temps
    X_norm_train, X_norm_temp = train_test_split(
        X_norm, test_size=0.4, random_state=random_state, shuffle=True
    )  # 60% train, 40% restant

    # Sur les 40% restants, on veut 30/10 (75%/25%)
    X_norm_test, X_norm_val = train_test_split(
        X_norm_temp, test_size=0.25, random_state=random_state, shuffle=True
    )

    # Pour les anomalies : 80% test, 20% val
    X_anom_test, X_anom_val = train_test_split(
        X_anom, test_size=0.2, random_state=random_state, shuffle=True
    )

    # Construction des sets finaux
    X_train = X_norm_train  # uniquement normal
    y_train = np.zeros(len(X_train), dtype=int)

    X_test = np.vstack([X_norm_test, X_anom_test])
    y_test = np.concatenate([
        np.zeros(len(X_norm_test), dtype=int),
        np.ones(len(X_anom_test), dtype=int),
    ])

    X_val = np.vstack([X_norm_val, X_anom_val])
    y_val = np.concatenate([
        np.zeros(len(X_norm_val), dtype=int),
        np.ones(len(X_anom_val), dtype=int),
    ])

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
def normalize_ecg_splits():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_ecg_train_val_test()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test), scaler


# IFT799 — Clustering & Détection d'anomalies (Python / ML)
**Université de Sherbrooke — Automne 2025**  
Cours : Sciences des données (IFT799)

> Apprentissage non supervisé appliqué à deux domaines biomédicaux : clustering de données génomiques (Hi-Seq) et détection d'anomalies dans des signaux ECG cardiaques.

---

## 🧬 Aperçu du projet

| Partie | Données | Méthodes | Métriques |
|--------|---------|----------|-----------|
| **Clustering** | Hi-Seq (expression génique, 5 cancers) | K-Means, Spectral Clustering, DBSCAN | ARI, NMI, Silhouette |
| **Détection d'anomalies** | ECG (battements cardiaques) | Isolation Forest, Auto-encodeur simple, Auto-encodeur profond | AUC, F1, Précision, Rappel |

---

## 📁 Structure du projet

```
IFT799-ML-Clustering-Anomaly/
├── src/
│   ├── models.py            # Algorithmes de clustering (K-Means, DBSCAN, Spectral)
│   ├── models_anomaly.py    # Modèles de détection (Isolation Forest, Auto-encodeurs)
│   ├── preprocess.py        # Prétraitement, normalisation, PCA, UMAP
│   ├── train.py             # Pipeline d'entraînement
│   └── utils.py             # Fonctions utilitaires
├── notebooks/
│   └── TP2_main.ipynb       # Notebook principal — expériences et visualisations
├── reports/
│   └── Rapport_TP2_IFT799.pdf
└── .gitignore
```

---

## 🔬 Partie 1 — Clustering génomique Hi-Seq

### Données
- Vecteurs d'expression génique de patients atteints de **5 types de cancer** :
  `BRCA`, `LUAD`, `COAD`, `KIRC`, `PRAD`
- Espace de très haute dimension → réduction nécessaire

### Pipeline
```
Données brutes → Normalisation Z-score → PCA / UMAP → Clustering → Évaluation
```

### Méthodes
- **K-Means** — partitionnement en 5 clusters, exécuté dans 3 espaces (brut, PCA, UMAP)
- **Spectral Clustering** — exploite la structure du graphe de voisinage (affinity = nearest_neighbors)
- **DBSCAN** — basé densité, détecte automatiquement les points aberrants

### Réduction de dimension
| Méthode | Type | Dimensions | Usage |
|---------|------|------------|-------|
| PCA | Linéaire | 100 composantes | K-Means, DBSCAN |
| UMAP | Non-linéaire | 100 dimensions | Meilleure séparabilité biologique |

### Résultats (extrait)
| Algorithme | Espace | ARI | Silhouette |
|-----------|--------|-----|------------|
| Spectral | Full | 0.659 | 0.091 |
| K-Means | Full | 0.697 | 0.111 |
| DBSCAN | Full | 0.000 | NaN |

---

## 🫀 Partie 2 — Détection d'anomalies ECG

### Données
- Signaux ECG segmentés en battements individuels
- Dataset fortement déséquilibré (majorité de signaux normaux)
- Objectif : détecter les **arythmies** (fibrillations, extra-systoles)

### Stratégie
Les auto-encodeurs sont entraînés **uniquement sur les signaux normaux**. Un signal anormal génère une erreur de reconstruction élevée → seuil optimal déterminé par **maximisation du F1-score** sur validation.

### Architectures

**Auto-encodeur simple** (`build_autoencoder`)
```
Input → Dense(64, relu) → Dropout(0.3) → Dense(32, relu)
      → Dense(64, relu) → Dropout(0.3) → Output
```

**Auto-encodeur profond** (`build_deep_autoencoder`)
```
Input → GaussianNoise(0.1) → Dense(32, relu) → Dropout(0.4) → Dense(16, relu)
      → Dense(32, relu) → Dropout(0.4) → Output
```

### Résultats comparatifs
| Modèle | AUC | F1 | Précision | Rappel |
|--------|-----|----|-----------|--------|
| Isolation Forest | 0.946 | 0.221 | **0.927** | 0.125 |
| Auto-encodeur simple | **0.955** | 0.364 | 0.965 | 0.224 |
| Auto-encodeur profond | 0.953 | **0.388** | 0.963 | **0.243** |

**Conclusion** : Les auto-encodeurs surpassent Isolation Forest. En contexte clinique, l'AE profond est préférable (maximise le rappel = détecte plus d'anomalies cardiaques).

---

## 🛠️ Technologies

```python
# ML / Data Science
scikit-learn    # K-Means, DBSCAN, Spectral, Isolation Forest, métriques
tensorflow/keras # Auto-encodeurs (Dense, Dropout, GaussianNoise)
numpy, pandas   # Manipulation de données
umap-learn      # Réduction de dimension non-linéaire

# Visualisation
matplotlib, seaborn  # Courbes ROC, histogrammes, PCA 2D
```

---

## ▶️ Installation et exécution

```bash
# Cloner le repo
git clone https://github.com/CarlosTsambou/IFT799-ML-Clustering-Anomaly.git
cd IFT799-ML-Clustering-Anomaly

# Installer les dépendances
pip install scikit-learn tensorflow numpy pandas matplotlib seaborn umap-learn

# Lancer le notebook
jupyter notebook notebooks/TP2_main.ipynb
```

> **Note** : Les données Hi-Seq et ECG ne sont pas incluses dans ce repo (taille volumineuse). Elles sont disponibles sur les plateformes publiques TCGA et PhysioNet.



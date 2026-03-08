import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
EXPERIMENT_NAME = "Exp_05_Landing_VRTG_Analysis_Prediction_ALL_FEATURES_touch_included"
HARD_LANDING_THRESHOLD = 1.35
HARD_LANDING_WEIGHT = 3.0

# Configuration du Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Gestion des chemins
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path: sys.path.append(str(project_root))

try:
    from src.config import DATASETS_DIR
except ImportError:
    # Fallback pour exécution standalone
    DATASETS_DIR = Path("C:/Users/odino/PycharmProjects/AI_ALPHA_2/data/output/datasets")

DATASET_PATH = DATASETS_DIR / EXPERIMENT_NAME / "dataset_dense.npz"


# =============================================================================
# 2. FONCTIONS D'AGRÉGATION (FEATURE ENGINEERING)
# =============================================================================
def compute_aggregated_features(X_3d, mask=None):
    """
    Transforme un tenseur 3D (N, Time, Feat) en matrice 2D (N, Feat * 5).
    Extrait: Mean, Std, Min, Max, Final_Value pour chaque capteur.
    """
    N, T, F = X_3d.shape
    aggregated_features = []

    logger.info(f"Agrégation statistique de {F} capteurs...")

    for i in range(F):
        sensor_data = X_3d[:, :, i]  # (N, T)
        stats_per_sample = []

        for j in range(N):
            if mask is not None:
                # Utilise le masque pour ne prendre que les vrais pas de temps
                valid_steps = sensor_data[j][mask[j]]
            else:
                # Fallback: ignore les zéros stricts (padding)
                raw_steps = sensor_data[j]
                valid_steps = raw_steps[raw_steps != 0] if np.any(raw_steps != 0) else np.array([0.0])

            if len(valid_steps) == 0: valid_steps = np.array([0.0])

            # Calcul des statistiques clés
            stats = [
                np.mean(valid_steps),
                np.std(valid_steps),
                np.min(valid_steps),  # Important pour IVV (Descente Max)
                np.max(valid_steps),
                valid_steps[-1]  # Valeur à l'impact (si aligné)
            ]
            stats_per_sample.append(stats)

        aggregated_features.append(np.array(stats_per_sample))

    # Empilement horizontal : (N, F*5)
    X_agg = np.hstack(aggregated_features)
    return X_agg


# =============================================================================
# 3. PIPELINE PRINCIPAL
# =============================================================================
def main():
    logger.info("--- RANDOM FOREST GRID SEARCH SUR DONNÉES AGRÉGÉES ---")

    # 1. Chargement
    if not DATASET_PATH.exists():
        logger.error(f"Dataset introuvable : {DATASET_PATH}")
        return

    data = np.load(DATASET_PATH, allow_pickle=True)
    X_train_raw = data['X_train']
    X_val_raw = data['X_val']
    y_train = data['y_train']
    y_val = data['y_val']

    mask_train = data['mask_train'].astype(bool) if 'mask_train' in data else None
    mask_val = data['mask_val'].astype(bool) if 'mask_val' in data else None

    logger.info(f"Input Raw Shape: {X_train_raw.shape}")

    # 2. Agrégation (Réduction de dimension)
    # Passe de (N, 456, 17) -> (N, 85)
    X_train_agg = compute_aggregated_features(X_train_raw, mask_train)
    X_val_agg = compute_aggregated_features(X_val_raw, mask_val)

    logger.info(f"Input Aggregated Shape: {X_train_agg.shape}")

    # 3. Standardisation
    # Important pour comparer les coefficients ou aider la convergence si on change de modèle
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_agg)
    X_val_scaled = scaler.transform(X_val_agg)

    # Standardisation de la Cible (Crucial pour le Benchmark MSE=1.0)
    y_mean, y_std = np.mean(y_train), np.std(y_train)
    y_train_scaled = (y_train - y_mean) / y_std
    y_val_scaled = (y_val - y_mean) / y_std

    # Poids pour les Hard Landings
    sample_weights = np.where(y_train >= HARD_LANDING_THRESHOLD, HARD_LANDING_WEIGHT, 1.0).flatten()

    # 4. Configuration de la Grille de Recherche
    # On explore la complexité : de modèles très simples (sous-apprentissage) à complexes
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],  # Contrôle la capacité à mémoriser
        'min_samples_leaf': [1, 2, 5, 10],  # Contrôle le lissage du bruit
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2', 0.5]  # Force à regarder différents sous-ensembles de features
    }

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    # 5. Lancement de la Recherche
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=30,  # Teste 30 combinaisons aléatoires
        cv=3,  # Validation croisée 3 plis
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    logger.info("Lancement de l'optimisation des hyperparamètres (30 itérations)...")
    # Note: On passe les sample_weights via fit_params
    search.fit(X_train_scaled, y_train_scaled, sample_weight=sample_weights)

    best_model = search.best_estimator_
    logger.info(f"Meilleurs paramètres : {search.best_params_}")

    # 6. Évaluation Finale
    preds = best_model.predict(X_val_scaled)
    mse = mean_squared_error(y_val_scaled, preds)
    mae = mean_absolute_error(y_val_scaled, preds)

    logger.info("-" * 40)
    logger.info(f"RÉSULTAT FINAL (MSE sur Cible Scalée)")
    logger.info(f"Validation MSE: {mse:.6f} (Benchmark Dummy: 1.0)")
    logger.info(f"Validation MAE: {mae:.6f}")
    logger.info("-" * 40)

    # 7. Interprétation Automatique
    if mse < 0.90:
        logger.info(">>> SUCCÈS MAJEUR : Signal fort détecté !")
        logger.info("    Le problème est résolu. Vous pouvez affiner ce modèle.")
    elif mse < 0.98:
        logger.info(">>> SUCCÈS MODÉRÉ : Signal détecté.")
        logger.info("    L'agrégation fonctionne mieux que les données brutes.")
    elif mse < 1.0:
        logger.info(">>> LIMITE : Signal très faible.")
    else:
        logger.info(">>> ÉCHEC : Même avec agrégation et optimisation, aucun signal.")
        logger.info("    Si l'analyse forensique (script 08) était bonne, essayez d'ajouter")
        logger.info("    des features fréquentielles (FFT) ou changez les capteurs.")


if __name__ == "__main__":
    main()
import sys
from pathlib import Path

import numpy as np

# Ajout du root project au path pour les imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.config import PROCESSED_DIR, DATA_DIR, SELECTED_PARAMS_FILE
from src.processing.building.builder import DatasetBuilder


def load_features_from_config(target_name: str) -> list[str]:
    """
    Charge la liste des paramètres depuis le fichier texte externe défini dans config.py.
    Applique un filtrage pour exclure la variable cible et garantir l'intégrité des données.
    """
    if not SELECTED_PARAMS_FILE.exists():
        raise FileNotFoundError(
            f"[ERR] Fichier de configuration des paramètres introuvable : {SELECTED_PARAMS_FILE}\n"
            f"      Veuillez vérifier qu'il existe bien dans le dossier 'data/'."
        )

    print(f"[INFO] Chargement dynamique des paramètres depuis : {SELECTED_PARAMS_FILE.name}")

    with open(SELECTED_PARAMS_FILE, 'r') as f:
        # Nettoyage : suppression des espaces, sauts de ligne, et lignes vides
        params = [line.strip() for line in f if line.strip()]

    # Validation : La target ne doit pas être dans les features (Data Leakage prevention)
    features = [p for p in params if p != target_name]

    if len(features) == 0:
        raise ValueError("[ERR] La liste des features est vide après filtrage !")

    print(f"[INFO] {len(features)} descripteurs (features) chargés. (Cible '{target_name}' exclue).")
    return features


def test_dataset_building_real_conditions():
    print("========================================")
    print("   TESTING DATASET BUILDER (REAL COND)  ")
    print("========================================")

    # 1. Configuration des Répertoires
    # Adaptez 'input_dir' selon où se trouvent vos fichiers HDF5 traités (ex: test_output ou by_phase)
    input_dir = PROCESSED_DIR / "test_output"
    output_dir = DATA_DIR / "dataset_full_test"

    print(f"[INFO] Input Directory: {input_dir}")
    print(f"[INFO] Output Directory: {output_dir}")

    builder = DatasetBuilder(input_dir=input_dir, output_dir=output_dir)

    # 2. Définition des Conditions Réelles
    target = "VRTG_C"  # Accélération verticale au toucher (variable critique)
    phases = [10, 11, 12]  # Phase d'atterrissage (Approche, Arrondi, Roulage)

    try:
        # 3. Chargement Dynamique (Option 2)
        features = load_features_from_config(target)

        # 4. Exécution du Builder
        print(f"[INFO] Lancement du build pour phases {phases}...")
        datasets = builder.build_dataset(
            features=features,
            target=target,
            phases=phases,
            test_size=0.2,
            val_size=0.2,
            random_state=42
        )

        # 5. Validation Dimensionnelle des Tenseurs
        print("\n--- Validation des Dimensions (Tenseurs Denses) ---")
        for split in ['train', 'val', 'test']:
            key_X = f"X_{split}"
            key_y = f"y_{split}"

            # Vérification de l'existence des clés (gestion du cas "dataset trop petit")
            if key_X not in datasets:
                print(f"[WARN] Split '{split}' absent du résultat.")
                continue

            X = datasets[key_X]
            y = datasets[key_y]
            mask = datasets[f"mask_{split}"]

            print(f"Split: {split.upper()}")
            print(f"  - N Samples : {X.shape[0]}")

            if X.shape[0] > 0:
                print(f"  - X Shape   : {X.shape}  (Samples, Time, Features)")
                print(f"  - y Shape   : {y.shape}  (Samples,)")
                print(f"  - Mask Shape: {mask.shape}")

                # Vérification de cohérence mathématique
                if X.shape[2] != len(features):
                    print(f"  [ALERT] Incohérence Features: Attendu {len(features)}, Reçu {X.shape[2]}")

                # Vérification de la qualité des données (NaNs)
                if np.isnan(X).any():
                    print("  [ALERT] Présence de NaN dans X (Padding mal géré ou données corrompues) !")
                else:
                    print("  [OK] Pas de NaN détectés.")

    except Exception as e:
        print(f"\n[FAIL] Échec du processus : {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_dataset_building_real_conditions()
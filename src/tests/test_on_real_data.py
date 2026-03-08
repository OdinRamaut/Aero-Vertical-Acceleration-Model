import sys
from pathlib import Path

import h5py

# Setup du path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import des nouveaux chemins configurés (Architecture Input/Output)
from src.config import (
    SIMPLIFIED_DIR,
    PROCESSED_DIR,
    DATASETS_DIR,
    SELECTED_PARAMS_FILE
)
from src.processing.resampling import FlightDataTransformer
from src.processing.features.engineering import FeatureEngineer
from src.processing.building.builder import DatasetBuilder


def autodiscover_features(lake_dir: Path, target_name: str) -> list[str]:
    """
    Scanne le premier fichier du Data Lake pour trouver TOUS les paramètres disponibles.
    Utilisé quand aucun fichier de configuration n'est fourni.
    """
    files = list(lake_dir.glob("*.h5")) + list(lake_dir.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"Data Lake vide : {lake_dir}")

    ref_file = files[0]
    print(f"[AUTO-DISCOVERY] Scan des paramètres sur : {ref_file.name}")

    with h5py.File(ref_file, 'r') as f:
        if 'Parameters' not in f:
            raise ValueError("Groupe 'Parameters' introuvable.")

        # On prend tout sauf Time, Mask et la Cible
        all_params = []
        for p in f['Parameters'].keys():
            if p not in ["Time", "Mask", "FLIGHT_PHASE", target_name]:
                all_params.append(p)

    print(f"[AUTO-DISCOVERY] {len(all_params)} paramètres trouvés.")
    return sorted(all_params)


def load_features_config(target_name: str, config_file: Path):
    """Charge une configuration de features spécifique."""
    if not config_file.exists():
        return None

    with open(config_file, 'r') as f:
        params = [l.strip() for l in f if l.strip() and l.strip() != "MetaData" and l.strip() != target_name]
    return params


def run_full_batch_pipeline():
    print("========================================")
    print("   PIPELINE COMPLET (STRUCTURE INPUT/OUTPUT) ")
    print("========================================")

    # 1. DÉFINITION DU DATA LAKE (Zone Output Read-Write)
    # PROCESSED_DIR pointe déjà vers data/output/processed_lake dans config.py
    processed_lake_dir = PROCESSED_DIR
    processed_lake_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Input Raw Dir  : {SIMPLIFIED_DIR}")
    print(f"[INFO] Data Lake Dir  : {processed_lake_dir}")
    print(f"[INFO] Datasets Dir   : {DATASETS_DIR}")

    # 2. INGESTION MASSIVE (Zone Input Read-Only)
    source_files = list(SIMPLIFIED_DIR.glob("*.hdf5")) + list(SIMPLIFIED_DIR.glob("*.h5"))

    if not source_files:
        print(f"[ERR] Aucun fichier source trouvé dans {SIMPLIFIED_DIR}")
        print("      Vérifiez que vous avez bien déplacé vos fichiers dans data/input/simplified_raw/")
        return

    print(f"\n[PHASE 1] Ingestion de {len(source_files)} fichiers...")

    transformer = FlightDataTransformer(target_freq=4)
    engineer = FeatureEngineer(sampling_freq=4.0)

    successful_files = 0

    for i, input_file in enumerate(source_files, 1):
        filename = input_file.name
        lake_file = processed_lake_dir / f"processed_{filename}"

        # Optionnel : skip si déjà traité
        # if lake_file.exists():
        #     successful_files += 1
        #     continue

        print(f"--- Traitement {i}/{len(source_files)} : {filename} ---")
        try:
            # A. Transformation (Output dans processed_lake)
            if lake_file.exists(): lake_file.unlink()

            output_path = transformer.process_flight(input_file, lake_file)

            if not output_path:
                print(f"  [SKIP] Transformation échouée.")
                continue

            # B. Enrichissement (In-Place dans processed_lake)
            engineer.process_file(lake_file)
            print(f"  [OK] -> {lake_file.name}")
            successful_files += 1

        except Exception as e:
            print(f"  [FAIL] Erreur : {e}")

    if successful_files == 0:
        print("[STOP] Aucun fichier valide généré.")
        return

    # 3. CONSTRUCTION DU DATASET
    print("\n[PHASE 2] Construction du Dataset d'Entraînement")

    target = "VRTG_C"
    phases = [10, 11, 12]

    # Chargement Config ou Auto-Découverte
    features_list = load_features_config(target, SELECTED_PARAMS_FILE)

    if features_list is None:
        print("[INFO] Mode AUTO-DÉCOUVERTE (fichier params absent).")
        try:
            features_list = autodiscover_features(processed_lake_dir, target)
        except Exception as e:
            print(f"[FAIL] Auto-découverte impossible : {e}")
            return
    else:
        print(f"[INFO] Utilisation de la liste configurée ({len(features_list)} features).")

    try:
        # On crée un sous-dossier spécifique pour ce run de test dans data/output/datasets
        run_output_dir = DATASETS_DIR / "test_run_global"

        builder = DatasetBuilder(input_dir=processed_lake_dir, output_dir=run_output_dir)

        datasets = builder.build_dataset(
            features=features_list,
            target=target,
            phases=phases,
            test_size=0.2,
            val_size=0.2
        )

        print("\n--- Validation Finale ---")
        print(f"Dataset sauvegardé dans : {run_output_dir}")
        print(f"X_train shape : {datasets['X_train'].shape}")

    except Exception as e:
        print(f"[FAIL] Construction dataset : {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_full_batch_pipeline()
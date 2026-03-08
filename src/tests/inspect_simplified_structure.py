import sys
from pathlib import Path

import h5py

# 1. Configuration du PYTHONPATH pour importer src.config
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

#from src.config import SIMPLIFIED_DIR


SIMPLIFIED_DIR=Path(r'C:\Users\odino\PycharmProjects\AI_ALPHA_2\data\input\by_PHASE')

def print_structure(name, obj):
    """
    Fonction de rappel pour h5py.visititems.
    Affiche les détails de chaque nœud (Groupe ou Dataset).
    """
    indent = "  " * (name.count("/") + 1)

    if isinstance(obj, h5py.Group):
        print(f"{indent}[G] {name}/")
    elif isinstance(obj, h5py.Dataset):
        # Affiche la forme (Time, Features) ou (Time,) et le type
        print(f"{indent}[D] {name} {obj.shape} ({obj.dtype})")

    # Affichage des attributs (métadonnées) s'ils existent
    if obj.attrs:
        for k, v in obj.attrs.items():
            # Limite l'affichage des valeurs trop longues pour la lisibilité
            val_str = str(v)
            if len(val_str) > 50:
                val_str = val_str[:47] + "..."
            print(f"{indent}    - @{k}: {val_str}")


def inspect_simplified_file():
    print(f"Recherche dans : {SIMPLIFIED_DIR}")

    if not SIMPLIFIED_DIR.exists():
        print(f"[ERREUR] Le répertoire {SIMPLIFIED_DIR} n'existe pas.")
        return

    files = list(SIMPLIFIED_DIR.glob("*.h5")) + list(SIMPLIFIED_DIR.glob("*.hdf5"))

    if not files:
        print("[ERREUR] Aucun fichier HDF5 trouvé.")
        return

    target_file = files[0]
    print(f"\n=== Analyse Structurelle de : {target_file.name} ===\n")

    try:
        with h5py.File(target_file, 'r') as f:
            # Affiche les attributs racines
            if f.attrs:
                print("Attributs Racine :")
                for k, v in f.attrs.items():
                    print(f"  @{k}: {v}")
                print("-" * 40)

            # Parcours récursif de tout le fichier
            f.visititems(print_structure)

    except Exception as e:
        print(f"[CRITIQUE] Impossible de lire le fichier : {e}")


if __name__ == "__main__":
    inspect_simplified_file()
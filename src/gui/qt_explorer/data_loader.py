from pathlib import Path
from typing import Dict, List, Any

import numpy as np


class DatasetLoader:
    """
    Handles loading and parsing of .npz dataset files.
    Decoupled from UI logic.
    """

    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.target_name: str = "Unknown"
        self.ids_cache: Dict[str, str] = {}
        self.filename: str = ""

    def load_file(self, file_path: str) -> bool:
        """
        Loads the .npz file and extracts metadata.
        Returns True if successful, raises Exception on failure.
        """
        try:
            # allow_pickle=True est souvent nécessaire pour les métadonnées string
            self.data = np.load(file_path, allow_pickle=True)
            self.filename = Path(file_path).name
            self.ids_cache = {}  # Clear cache on new load

            # Parse Feature Names
            if 'feature_names' in self.data:
                raw_feats = self.data['feature_names']
                self.feature_names = [
                    f.decode('utf-8') if isinstance(f, bytes) else str(f)
                    for f in raw_feats
                ]
            elif 'X_train' in self.data:
                # Fallback: Generate generic names if missing
                n_feats = self.data['X_train'].shape[2]
                self.feature_names = [f"Feat_{i}" for i in range(n_feats)]
            else:
                self.feature_names = []

            # Parse Target Name
            if 'target_name' in self.data:
                t = self.data['target_name']
                if isinstance(t, (list, np.ndarray)) and len(t) > 0:
                    self.target_name = t[0].decode('utf-8') if isinstance(t[0], bytes) else str(t[0])
                else:
                    self.target_name = str(t)
            else:
                self.target_name = "Target"

            return True

        except Exception as e:
            raise IOError(f"Failed to load dataset: {str(e)}")

    def get_flight_id(self, split: str, idx: int) -> str:
        """Retrieves or decodes the Flight ID for a given index with caching."""
        cache_key = f"{split}_{idx}"
        if cache_key in self.ids_cache:
            return self.ids_cache[cache_key]

        id_key = f"ids_{split}"
        if id_key in self.data:
            raw_id = self.data[id_key][idx]
            flight_id = raw_id.decode('utf-8') if isinstance(raw_id, bytes) else str(raw_id)
        else:
            flight_id = f"Sample_{idx}"

        self.ids_cache[cache_key] = flight_id
        return flight_id

    def get_splits(self) -> List[str]:
        """Returns a list of available splits (train/val/test)."""
        splits = []
        for s in ['train', 'val', 'test']:
            if f"X_{s}" in self.data:
                splits.append(s)
        return splits
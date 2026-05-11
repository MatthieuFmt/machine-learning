"""Résolution des chemins au runtime — aucun hardcoding.

Les chemins sont construits à partir d'un répertoire racine (par défaut le
working directory) et du nom de l'instrument.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class PathConfig:
    """Tous les chemins du pipeline, résolus relativement à root_dir.

    Exemple :
        >>> pc = PathConfig(root_dir='./project')
        >>> pc.dir_ready
        WindowsPath('project/ready-data')
    """

    root_dir: str = "."

    # Sous-dossiers
    dir_raw: str = "data"
    dir_clean: str = "cleaned-data"
    dir_ready: str = "ready-data"
    dir_results: str = "results"
    dir_predictions: str = "predictions"

    @property
    def root(self) -> Path:
        return Path(self.root_dir).resolve()

    @property
    def raw(self) -> Path:
        return self.root / self.dir_raw

    @property
    def clean(self) -> Path:
        return self.root / self.dir_clean

    @property
    def ready(self) -> Path:
        return self.root / self.dir_ready

    @property
    def results(self) -> Path:
        return self.root / self.dir_results

    @property
    def predictions(self) -> Path:
        return self.root / self.dir_predictions

    def ensure_dirs(self) -> None:
        """Crée tous les répertoires de sortie s'ils n'existent pas."""
        for d in [self.clean, self.ready, self.results, self.predictions]:
            d.mkdir(parents=True, exist_ok=True)

    def clean_file(self, instrument: str, timeframe: str) -> Path:
        """Chemin vers un fichier nettoyé.

        >>> PathConfig().clean_file('EURUSD', 'H1').name
        'EURUSD_H1_cleaned.csv'
        """
        return self.clean / f"{instrument}_{timeframe}_cleaned.csv"

    def ml_ready_file(self, instrument: str) -> Path:
        """Chemin vers le CSV ML-ready.

        >>> PathConfig().ml_ready_file('EURUSD').name
        'EURUSD_Master_ML_Ready.csv'
        """
        return self.ready / f"{instrument}_Master_ML_Ready.csv"

    def predictions_file(self, instrument: str, year: int) -> Path:
        """Chemin vers les prédictions d'une année.

        >>> PathConfig().predictions_file('EURUSD', 2024).name
        'Predictions_2024_TripleBarrier.csv'
        """
        return self.results / f"Predictions_{year}_TripleBarrier.csv"

    def trades_detailed_file(self, instrument: str, year: int) -> Path:
        """Chemin vers les trades détaillés."""
        return self.results / f"Trades_Detailed_{year}.csv"

    def feature_importance_file(self, instrument: str, train_end_year: int) -> Path:
        """Chemin vers le CSV d'importance des features."""
        return self.results / f"Feature_Importance_train{train_end_year}.csv"

    def report_md_file(
        self, instrument: str, year: int, version: str | None = None
    ) -> Path:
        """Chemin vers le rapport Markdown.

        Si `version` est fourni, le rapport est dans un sous-dossier.
        """
        base = self.predictions
        if version:
            base = base / version
        return base / f"Rapport_Performance_{year}.md"

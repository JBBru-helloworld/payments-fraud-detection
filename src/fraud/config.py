"""Project-wide constants: seed, paths, split fractions and cost assumptions."""

from pathlib import Path

RANDOM_SEED: int = 42

ROOT_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

TRAIN_TRANSACTION_CSV: Path = RAW_DIR / "train_transaction.csv"
TRAIN_IDENTITY_CSV: Path = RAW_DIR / "train_identity.csv"
MERGED_PARQUET: Path = INTERIM_DIR / "train_merged.parquet"

# Chronological split on TransactionDT: train, validation, test.
SPLIT_FRACTIONS: tuple[float, float, float] = (0.70, 0.15, 0.15)

# Filled in Phase 5.
COSTS: dict[str, float] = {}

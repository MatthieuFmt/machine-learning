---
name: logging-structured-ml
description: Logging structuré pour pipeline ML — fichier + console, rotation, shape tracking, erreurs tracées.
---

# Logging Structuré ML — Traçabilité Complète

## Règle cardinale

**Toute fonction publique du pipeline ML doit logger ENTRY, EXIT, et toute transformation de DataFrame (shape avant/après, % de perte). Les `print()` sont interdits. Le logging doit écrire dans un fichier avec rotation ET sur stderr.**

## Instructions

### 1. Configuration fichier + console obligatoire

```python
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_file_logging(
    log_dir: str = "logs",
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> None:
    """Configure le logging racine avec rotation fichier + console."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # Format structuré
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler fichier avec rotation
    file_handler = RotatingFileHandler(
        Path(log_dir) / "pipeline.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level)

    # Handler erreurs séparé
    error_handler = RotatingFileHandler(
        Path(log_dir) / "errors.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    error_handler.setFormatter(fmt)
    error_handler.setLevel(logging.ERROR)

    # Handler console
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(level)

    # Nettoyage des handlers existants (évite doublons)
    root.handlers.clear()

    root.addHandler(file_handler)
    root.addHandler(error_handler)
    root.addHandler(console_handler)
```

### 2. Pattern ENTRY/EXIT obligatoire

```python
import time
from learning_machine_learning.core.logging import get_logger

logger = get_logger(__name__)


def fetch_ohlcv(exchange, symbol, timeframe, since, limit):
    t0 = time.perf_counter()
    logger.info("ENTRY | fetch_ohlcv | exchange=%s | symbol=%s | tf=%s | limit=%d",
                exchange.id, symbol, timeframe, limit)

    try:
        # ... core logic ...

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("EXIT | fetch_ohlcv | %d rows | %.1f ms | %s → %s",
                    len(df), elapsed_ms, df.index[0], df.index[-1])
        return df

    except Exception:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.exception("FATAL | fetch_ohlcv | failed after %.1f ms", elapsed_ms)
        raise
```

### 3. Shape tracking systématique

```python
def log_row_loss(step_name: str, n_before: int, n_after: int) -> None:
    """Logge la perte de lignes d'une étape de filtrage."""
    lost = n_before - n_after
    pct = lost / n_before * 100 if n_before else 0
    logger.info("SHAPE | %s | %d → %d rows (%.1f%% loss)", step_name, n_before, n_after, pct)


# Usage
n_before = len(df)
df.dropna(subset=["Target"], inplace=True)
log_row_loss("dropna Target", n_before, len(df))
```

### 4. Logging des NaN et des colonnes manquantes

```python
def log_nan_report(df: pd.DataFrame, stage: str) -> None:
    """Logge un rapport des NaN par colonne."""
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if nan_cols.empty:
        logger.debug("SHAPE | %s | 0 NaN", stage)
    else:
        for col, count in nan_cols.items():
            pct = count / len(df) * 100
            logger.warning("SHAPE | %s | NaN in '%s': %d/%d (%.1f%%)",
                          stage, col, count, len(df), pct)
```

### 5. Niveaux de log par contexte

```python
# DEBUG   : Forme des DataFrames, valeurs intermédiaires
# INFO    : ENTRY/EXIT de fonction, étapes du pipeline
# WARNING : NaN détectés, données manquantes, métriques dégradées
# ERROR   : Échec d'une opération (retry possible), look-ahead détecté
# CRITICAL: Corruption de données, échec irrécupérable

# Exemples :
logger.debug("X_train dtypes: %s", X_train.dtypes.to_dict())
logger.info("ENTRY | train_model | n_samples=%d | n_features=%d", len(X_train), X_train.shape[1])
logger.warning("NaN détectés dans 'Close': %d/%d", nan_count, len(df))
logger.error("RateLimitExceeded — retry 2/3 in 4s")
logger.critical("Fichier corrompu: %s — SHA256 mismatch", filepath)
```

### 6. Tracing d'erreurs avec contexte

```python
# ✅ CORRECT : logger.exception() capture automatiquement le traceback
try:
    model.fit(X_train, y_train)
except Exception:
    logger.exception("FATAL | train_model | n_samples=%d | n_features=%d",
                     len(X_train), X_train.shape[1])
    raise ModelError("Échec entraînement") from None

# ❌ INCORRECT : logger.error() sans traceback
try:
    model.fit(X_train, y_train)
except Exception as e:
    logger.error("Échec: %s", e)  # pas de stack trace
```

## Checklist finale

1. `setup_file_logging()` appelé UNE fois au démarrage du pipeline
2. Fichier `logs/pipeline.log` + `logs/errors.log` avec rotation 10 MB
3. Chaque fonction publique a ENTRY + EXIT avec timings
4. Chaque transformation logge shape avant/après
5. Les NaN sont loggés en WARNING avec colonne et count
6. `logger.exception()` dans tous les blocs `except`
7. Pas de `print()` dans le code de production

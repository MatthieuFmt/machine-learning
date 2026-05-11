---
name: ccxt-fetcher-patterns
description: CCXT Error Handling & Retry — patterns robustes pour fetch_ohlcv, gestion RateLimitExceeded, fallback multi-exchange.
---

# CCXT Fetcher Patterns — Data Ingestion Crypto

## Règle cardinale

**Tout appel à l'API CCXT DOIT être wrappé dans un retry exponentiel avec fallback. Les exchanges crypto sont instables par nature — le code doit survivre à des coupures de 30 secondes sans perdre de données.**

## Instructions

### 1. Import obligatoire des exceptions CCXT

```python
import ccxt
from ccxt import (
    NetworkError,        # DNS, timeout, connection refused
    ExchangeError,       # erreur générique exchange
    RateLimitExceeded,   # HTTP 429 ou limite custom
    BadSymbol,           # symbole invalide
    AuthenticationError, # clé API invalide
    InsufficientFunds,   # fonds insuffisants (non applicable en fetch)
)
```

### 2. Retry exponentiel standard

```python
import time
from learning_machine_learning.core.logging import get_logger

logger = get_logger(__name__)

def _retry_fetch(
    fetch_func,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> list:
    """Retry exponentiel sur erreur CCXT."""
    for attempt in range(1, max_retries + 1):
        try:
            result = fetch_func()
            logger.debug("CCXT fetch OK (attempt %d/%d)", attempt, max_retries)
            return result
        except RateLimitExceeded as e:
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "RateLimit attempt %d/%d — retry in %.1fs | %s",
                attempt, max_retries, delay, str(e)[:200],
            )
            time.sleep(delay)
        except NetworkError as e:
            delay = base_delay * (2 ** (attempt - 1))
            logger.error(
                "NetworkError attempt %d/%d — retry in %.1fs | %s",
                attempt, max_retries, delay, str(e)[:200],
            )
            time.sleep(delay)
        except ExchangeError as e:
            logger.exception("ExchangeError fatal — pas de retry")
            raise
    raise RuntimeError(f"Échec après {max_retries} tentatives")
```

### 3. fetch_ohlcv avec fallback multi-exchange

```python
def fetch_ohlcv_robust(
    primary_exchange: ccxt.Exchange,
    fallback_exchanges: list[ccxt.Exchange],
    symbol: str,
    timeframe: str = "1h",
    since: int | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    """Fetch OHLCV avec fallback sur exchanges secondaires."""
    exchanges = [primary_exchange] + fallback_exchanges

    for idx, exchange in enumerate(exchanges):
        try:
            raw = _retry_fetch(
                lambda: exchange.fetch_ohlcv(symbol, timeframe, since, limit),
                max_retries=2,  # moins de retries car on a des fallbacks
            )
            if not raw:
                logger.warning("%s a retourné []", exchange.id)
                continue

            df = pd.DataFrame(
                raw, columns=["Time", "Open", "High", "Low", "Close", "Volume"]
            )
            df["Time"] = pd.to_datetime(df["Time"], unit="ms", utc=True)
            df.set_index("Time", inplace=True)
            df = df.astype({
                "Open": "float64", "High": "float64",
                "Low": "float64", "Close": "float64", "Volume": "float64",
            })

            logger.info(
                "fetch_ohlcv OK: %s/%s/%s via %s — %d bougies [%s → %s]",
                symbol, timeframe, exchange.id,
                "primary" if idx == 0 else "fallback",
                len(df), df.index[0], df.index[-1],
            )
            return df

        except BadSymbol:
            logger.error("BadSymbol: %s sur %s", symbol, exchange.id)
            continue
        except Exception:
            logger.exception("Échec %s — tentative exchange suivant", exchange.id)
            continue

    raise RuntimeError(f"Aucun exchange n'a pu fournir {symbol}/{timeframe}")
```

### 4. Cache d'exchange (réutilisation connexion)

```python
_exchange_cache: dict[str, ccxt.Exchange] = {}

def get_exchange(exchange_id: str = "binance") -> ccxt.Exchange:
    """Retourne une instance ccxt.Exchange cachée."""
    if exchange_id not in _exchange_cache:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({"enableRateLimit": True})
        exchange.load_markets()
        _exchange_cache[exchange_id] = exchange
        logger.info("Exchange %s initialisé (%d marchés)", exchange_id, len(exchange.markets))
    return _exchange_cache[exchange_id]
```

### 5. Conversion early float64

```python
# ✅ CORRECT : astype immédiat après ingestion
df = df.astype({"Open": "float64", "High": "float64", "Low": "float64",
                "Close": "float64", "Volume": "float64"})

# ❌ INCORRECT : laisser pandas inférer — coûteux, risque de object dtype
df["Close"] = df["Close"]  # dtype peut être object → conversions implicites coûteuses
```

## Contrôles automatiques

À chaque fetch, vérifier :

1. `assert len(df) > 0` — exchange n'a pas retourné de liste vide
2. `assert df.index.is_monotonic_increasing` — vérifier l'ordre chronologique
3. `assert not df.isna().any().any()` — pas de NaN dans l'OHLCV brut
4. Logger le temps écoulé : `logger.info("fetch %s %s: %d rows in %.1fs", symbol, timeframe, len(df), elapsed)`

"""Test du look-ahead bias sur les features D1 via merge_asof.

Le script recalcule RSI_14 a partir des D1 closes, puis simule le merge_asof
bugge (D1 du jour meme disponible) vs corrige (D1 de la veille uniquement).
"""
import pandas as pd
import numpy as np


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


d1 = pd.read_csv("cleaned-data/EURUSD_D1_cleaned.csv", index_col=0, parse_dates=True)
h1 = pd.read_csv("cleaned-data/EURUSD_H1_cleaned.csv", index_col=0, parse_dates=True)

# Recalculer RSI D1
d1["RSI_14"] = rsi(d1["Close"], 14)

# Preparer pour merge_asof
h1_s = h1.sort_index().reset_index()
d1_s = d1.sort_index().reset_index()

# D1 shifted: la D1 du jour J est disponible uniquement le jour J+1
d1_shifted = d1_s.copy()
d1_shifted["Time"] = d1_shifted["Time"] + pd.Timedelta(days=1)

# Merge bug vs corrige
m_bug = pd.merge_asof(h1_s, d1_s[["Time", "RSI_14"]], on="Time", direction="backward")
m_ok = pd.merge_asof(h1_s, d1_shifted[["Time", "RSI_14"]], on="Time", direction="backward")

rsi_bug = m_bug.set_index("Time")["RSI_14"]
rsi_ok = m_ok.set_index("Time")["RSI_14"]

# Forward returns
closes = h1["Close"]
fwd12 = closes.shift(-12) / closes - 1.0
fwd24 = closes.shift(-24) / closes - 1.0

print("=== Correlation RSI_14_D1 vs forward return ===")
for lbl, fwd in [("12h", fwd12), ("24h", fwd24)]:
    idx_b = rsi_bug.dropna().index.intersection(fwd.dropna().index)
    idx_o = rsi_ok.dropna().index.intersection(fwd.dropna().index)
    c_bug = np.corrcoef(rsi_bug.loc[idx_b], fwd.loc[idx_b])[0, 1]
    c_ok = np.corrcoef(rsi_ok.loc[idx_o], fwd.loc[idx_o])[0, 1]
    diff = c_bug - c_ok
    print(f"  Horizon {lbl}: bug={c_bug:+.4f}  correct={c_ok:+.4f}  delta={diff:+.4f}")

# Controle sur un jour precis
print()
print("=== 5 premieres barres H1 du 2024-01-03 ===")
subset_h1 = h1_s[(h1_s["Time"] >= "2024-01-03") & (h1_s["Time"] < "2024-01-04")].head(5)
subset_bug = pd.merge_asof(subset_h1, d1_s[["Time", "RSI_14"]], on="Time", direction="backward")
subset_ok = pd.merge_asof(
    subset_h1, d1_shifted[["Time", "RSI_14"]], on="Time", direction="backward"
)
comp = pd.DataFrame(
    {
        "H1_Time": subset_bug["Time"],
        "D1_bug_date": subset_bug["Time_y"],
        "D1_ok_date": subset_ok["Time_y"],
        "RSI_bug": subset_bug["RSI_14"],
        "RSI_ok": subset_ok["RSI_14"],
    }
)
print(comp.to_string(index=False))
print("bug : D1 du 03/01 disponible a 00:00 le 03/01 (look-ahead)")
print("ok  : D1 du 02/01 uniquement disponible")
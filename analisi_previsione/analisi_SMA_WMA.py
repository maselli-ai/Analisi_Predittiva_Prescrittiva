
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 1) Parametri e percorsi

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / "dataset_finale_ETL_QA.xlsx"  # <— adatto al tuo progetto; cambia se serve
OUT_DIR = HERE.parent / "output_311"                  # cartella out (coerente con tua struttura)
OUT_PLOTS = OUT_DIR / "grafici"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PLOTS.mkdir(parents=True, exist_ok=True)

WINDOW = 3
MONTH_ORDER = ["GENNAIO","FEBBRAIO","APRILE","MAGGIO","GIUGNO","LUGLIO","AGOSTO"]
MONTH_NUM = {m:i+1 for i,m in enumerate(MONTH_ORDER)}

# scelgo 2–3 articoli pilota (puoi aggiungere/togliere liberamente)
ARTICOLI_PILOTA = ["SLP3100", "SLISO180", "SLI749"]


# 2) Utilità metriche

def mae(y_true, y_pred):
    mask = (~y_true.isna()) & (~y_pred.isna())
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))

def rmse(y_true, y_pred):
    mask = (~y_true.isna()) & (~y_pred.isna())
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2)))

def mape(y_true, y_pred):
    mask = (~y_true.isna()) & (~y_pred.isna()) & (y_true != 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

# 3) Carico e preparo i dati
df = pd.read_excel(DATASET)

# normalizzo il mese e tengo solo il periodo target
df["mese_rif"] = df["mese_rif"].astype(str).str.upper().str.strip()
df = df[df["mese_rif"].isin(MONTH_ORDER)].copy()
df["mese_n"] = df["mese_rif"].map(MONTH_NUM)

# consumo = outgoing (numerico, NaN→0)
df["outgoing"] = pd.to_numeric(df["outgoing"], errors="coerce").fillna(0)

# serie mensile per articolo
serie = (df.groupby(["code","mese_rif","mese_n"], as_index=False)
           .agg(consumo=("outgoing","sum")))

# 4) Modelli SMA e WMA
def sma(y: pd.Series, window: int = 3) -> pd.Series:
    """Media mobile semplice."""
    return y.rolling(window=window).mean()

def wma(y: pd.Series, weights: np.ndarray) -> pd.Series:
    """Media mobile ponderata con pesi forniti (più peso ai dati recenti)."""
    w = np.array(weights, dtype=float)
    return y.rolling(window=len(w)).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

# 5) Loop sugli articoli pilota
rows = []  # qui accumulo le metriche per la tabella finale

for code in ARTICOLI_PILOTA:
    # estraggo serie dell'articolo ordinata per mese
    data = serie[serie["code"] == code].sort_values("mese_n")
    y = data["consumo"].reset_index(drop=True)
    mesi = data["mese_rif"].reset_index(drop=True)

    # PREVISIONE:
    # calcolo SMA(3) e WMA(3) e faccio shift(1) per prevedere il mese successivo
    sma_3 = sma(y, WINDOW).shift(1)
    wma_3 = wma(y, np.array([1, 2, 3])).shift(1)

    # METRICHE:
    mae_s = mae(y, sma_3); mape_s = mape(y, sma_3); rmse_s = rmse(y, sma_3)
    mae_w = mae(y, wma_3); mape_w = mape(y, wma_3); rmse_w = rmse(y, wma_3)

    # stampo a console (mi torna comodo per la discussione)
    print(f"\nArticolo: {code}")
    print(f"  SMA(3)  -> MAE={mae_s:.2f}, MAPE={mape_s:.2f}%, RMSE={rmse_s:.2f}")
    print(f"  WMA(3)  -> MAE={mae_w:.2f}, MAPE={mape_w:.2f}%, RMSE={rmse_w:.2f}")

    # salvo righe per tabella performance
    rows += [
        {"articolo": code, "modello": "SMA(3)", "MAE": mae_s, "MAPE(%)": mape_s, "RMSE": rmse_s},
        {"articolo": code, "modello": "WMA(3)", "MAE": mae_w, "MAPE(%)": mape_w, "RMSE": rmse_w},
    ]

    # GRAFICO confronto reale vs previsioni
    plt.figure(figsize=(9,5))
    plt.plot(mesi, y, marker="o", label="Domanda reale")
    plt.plot(mesi, sma_3, linestyle="--", marker="o", label="SMA(3)")
    plt.plot(mesi, wma_3, linestyle="--", marker="o", label="WMA(3)")
    plt.title(f"Confronto reale vs previsione – Articolo {code}")
    plt.xlabel("Mese")
    plt.ylabel("Consumo (unità)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / f"{code}_SMA_WMA.png", dpi=150)
    plt.close()

# 6) Tabella performance

perf = pd.DataFrame(rows)
perf = perf[["articolo","modello","MAE","MAPE(%)","RMSE"]]

# salvo sia in Excel che in CSV 
perf.to_excel(OUT_DIR / "SMA_WMA_performance.xlsx", index=False)
perf.to_csv(OUT_DIR / "SMA_WMA_performance.csv", index=False)

print("\nFile creati:")
print(" - Grafici:", OUT_PLOTS)
print(" - Tabella performance:", OUT_DIR / "SMA_WMA_performance.xlsx")


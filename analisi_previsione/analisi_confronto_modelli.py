import pandas as pd
from pathlib import Path

OUT_DIR = Path("../output_311")

# Importa le performance dei vari modelli
sma_wma = pd.read_excel(OUT_DIR / "SMA_WMA_performance.xlsx")
arima_prophet = pd.read_excel(OUT_DIR / "ARIMA_Prophet_performance.xlsx")

# Unisci in un’unica tabella
all_perf = pd.concat([sma_wma, arima_prophet], ignore_index=True)

# Calcola la media per modello
summary = all_perf.groupby("modello")[["MAE", "RMSE", "MAPE(%)"]].mean().round(2)
summary = summary.sort_values("MAPE(%)")

print(summary)
summary.to_excel(OUT_DIR / "Confronto_Modelli.xlsx")

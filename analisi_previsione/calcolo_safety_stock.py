# calcolo_safety_stock.py
# Commenti: codice per il calcolo della scorta di sicurezza (Safety Stock)

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Impostazioni di base
Z = 1.65  # livello di servizio 95%
OUT_DIR = Path("../output_311")
PLOTS_DIR = OUT_DIR / "grafici_prescrittivi"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Lettura file principali
tab = pd.read_excel("../output_311/Tabella_L_sigmaL_sigmaD.xlsx")
dem = pd.read_excel("../dataset_finale_ETL_QA.xlsx")

# Calcolo domanda media per articolo
media_domanda = (
    dem.groupby("code")["outgoing"]
       .mean()
       .reset_index()
       .rename(columns={"outgoing": "D_media"})
)

# Unione dei dati
tab = pd.merge(tab, media_domanda, on="code", how="left")

# Calcolo scorta di sicurezza
tab["SS"] = Z * np.sqrt(
    (tab["L_giorni"] * (tab["sigma_D_unita"] ** 2)) +
    ((tab["D_media"] ** 2) * (tab["sigma_L_giorni"] ** 2))
)

# Arrotondamento
tab["SS"] = tab["SS"].round(1)
tab["D_media"] = tab["D_media"].round(1)

# Esportazione tabella
fp_out = OUT_DIR / "Safety_Stock_per_articolo.xlsx"
tab.to_excel(fp_out, index=False)
print("Tabella creata:", fp_out)

# Grafico comparativo
plt.figure(figsize=(10, 6))
plt.scatter(tab["D_media"], tab["SS"])
plt.title("Scorta di sicurezza vs Domanda media")
plt.xlabel("Domanda media (unità)")
plt.ylabel("Scorta di sicurezza (unità)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "SafetyStock_vs_Demand.png", dpi=300, bbox_inches="tight")
plt.close()

print("Grafico salvato in:", PLOTS_DIR)

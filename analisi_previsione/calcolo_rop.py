# calcolo_rop.py

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

OUT_DIR = Path("../output_311")
PLOTS_DIR = OUT_DIR / "grafici_prescrittivi"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Articoli pilota opzionali; se vuoto, scelgo i primi 3 per ROP
PILOT_CODES = []  # es: ["SLSPC994", "SLGB390", "SLGCM816"]

# Input con L, sigma_L, sigma_D, D_media e SS
df = pd.read_excel(OUT_DIR / "Safety_Stock_per_articolo.xlsx")

# Definizioni: D = domanda media (unità/periodo), L = lead time medio (giorni)
# ROP = D * L + SS
df["ROP"] = (df["D_media"] * df["L_giorni"]) + df["SS"]

# Arrotondamenti
for c in ["L_giorni", "sigma_L_giorni", "sigma_D_unita", "D_media", "SS", "ROP"]:
    if c in df.columns:
        df[c] = df[c].round(2)

# Salvo tabella finale
fp_tab = OUT_DIR / "ROP_per_articolo.xlsx"
df.to_excel(fp_tab, index=False)
print("Tabella ROP creata:", fp_tab)

# Seleziono articoli pilota
if PILOT_CODES:
    pilots = df[df["code"].astype(str).isin(PILOT_CODES)].copy()
else:
    pilots = df.sort_values("ROP", ascending=False).head(3).copy()

# Grafico 1: barre ROP per i pilota
plt.figure(figsize=(9, 5))
plt.bar(pilots["code"].astype(str), pilots["ROP"])
plt.title("Reorder Point (ROP) - Articoli pilota")
plt.xlabel("Articolo")
plt.ylabel("ROP (unità)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ROP_pilota_bar.png", dpi=300, bbox_inches="tight")
plt.close()

# Grafico 2: scatter D vs ROP per tutti
plt.figure(figsize=(9, 6))
plt.scatter(df["D_media"], df["ROP"], s=18)
plt.title("Reorder Point vs Domanda media")
plt.xlabel("Domanda media (unità)")
plt.ylabel("ROP (unità)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ROP_vs_Demand.png", dpi=300, bbox_inches="tight")
plt.close()

print("Grafici salvati in:", PLOTS_DIR)


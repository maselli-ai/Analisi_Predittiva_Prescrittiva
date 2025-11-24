# analisi_as_is_to_be.py
# Confronto As-Is (simulato) vs To-Be usando solo i file già prodotti

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Cartelle output
OUT_DIR = Path("../output_311")
PLOTS_DIR = OUT_DIR / "grafici_prescrittivi"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Articoli pilota: se vuoto prendo i 3 con ROP_to_be più alto
PILOT_CODES: list[str] = []  # es: ["SLSPC994","SLGB390","SLGCM816"]

# File To-Be generato in precedenza (obbligatorio)
FILE_TO_BE = OUT_DIR / "ROP_per_articolo.xlsx"
if not FILE_TO_BE.exists():
    raise FileNotFoundError(f"Manca {FILE_TO_BE}. Esegui prima il calcolo di SS e ROP.")

# Carico To-Be e normalizzo colonne
to_be = pd.read_excel(FILE_TO_BE)
cols_needed = {"code","D_media","L_giorni","sigma_D_unita","sigma_L_giorni","SS","ROP"}
missing = cols_needed - set(to_be.columns)
if missing:
    raise KeyError(f"Mancano colonne nel To-Be: {sorted(missing)}. Presenti: {list(to_be.columns)}")

to_be = (to_be[["code","D_media","L_giorni","sigma_D_unita","sigma_L_giorni","SS","ROP"]]
         .rename(columns={"SS":"SS_to_be","ROP":"ROP_to_be"}))
to_be["code"] = to_be["code"].astype(str).str.strip()

# As-Is simulato (nessun file esterno) 
# Politica attuale semplificata: riordino a D*L, senza scorta di sicurezza
as_is = to_be[["code","D_media","L_giorni"]].copy()
as_is["ROP_as_is"] = as_is["D_media"] * as_is["L_giorni"]
as_is["SS_as_is"] = 0.0
as_is = as_is[["code","ROP_as_is","SS_as_is"]]

# Merge e indicatori
df = to_be.merge(as_is, on="code", how="inner").copy()
df["delta_ROP"]   = df["ROP_to_be"] - df["ROP_as_is"]
df["delta_ROP_%"] = np.where(df["ROP_as_is"] > 0, 100 * df["delta_ROP"] / df["ROP_as_is"], np.nan)
df["delta_SS"]    = df["SS_to_be"]  - df["SS_as_is"]

# Arrotondamenti
for c in ["D_media","L_giorni","sigma_D_unita","sigma_L_giorni",
          "SS_to_be","SS_as_is","ROP_to_be","ROP_as_is",
          "delta_ROP","delta_ROP_%","delta_SS"]:
    if c in df.columns:
        df[c] = df[c].round(2)

# Salvo tabella comparativa
fp_tab = OUT_DIR / "AsIs_ToBe_confronto.xlsx"
with pd.ExcelWriter(fp_tab, engine="xlsxwriter") as xw:
    df.to_excel(xw, sheet_name="Confronto", index=False)
print("Tabella comparativa creata:", fp_tab)

# Selezione articoli pilota
if PILOT_CODES:
    pilots = df[df["code"].astype(str).isin(PILOT_CODES)].copy()
    if pilots.empty:
        raise ValueError("I codici in PILOT_CODES non sono presenti nel dataset.")
else:
    pilots = df.sort_values("ROP_to_be", ascending=False).head(3).copy()

# Grafico 1: barre ROP As-Is vs To-Be 
plt.figure(figsize=(10,6))
idx = np.arange(len(pilots))
width = 0.38
plt.bar(idx - width/2, pilots["ROP_as_is"], width, label="As-Is")
plt.bar(idx + width/2, pilots["ROP_to_be"], width, label="To-Be")
plt.xticks(idx, pilots["code"].astype(str))
plt.ylabel("ROP (unità)")
plt.title("ROP As-Is vs To-Be – articoli pilota")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ROP_AsIs_vs_ToBe_pilota.png", dpi=300, bbox_inches="tight")
plt.close()

# Grafico 2: scomposizione ROP To-Be (D×L + SS)
plt.figure(figsize=(10,6))
comp_DL = pilots["D_media"] * pilots["L_giorni"]
comp_SS = pilots["SS_to_be"]
plt.bar(pilots["code"].astype(str), comp_DL, label="D × L")
plt.bar(pilots["code"].astype(str), comp_SS, bottom=comp_DL, label="SS")
plt.ylabel("ROP (unità)")
plt.title("Scomposizione ROP To-Be (D×L + SS) – articoli pilota")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ROP_ToBe_scomposizione_pilota.png", dpi=300, bbox_inches="tight")
plt.close()

# Grafico 3: variazione percentuale del ROP 
plt.figure(figsize=(10,6))
plt.bar(pilots["code"].astype(str), pilots["delta_ROP_%"])
plt.axhline(0, color="black", linewidth=0.8)
plt.ylabel("Δ ROP (%)")
plt.title("Variazione percentuale del ROP (To-Be vs As-Is)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ROP_delta_percent_pilota.png", dpi=300, bbox_inches="tight")
plt.close()

print("Grafici salvati in:", PLOTS_DIR)


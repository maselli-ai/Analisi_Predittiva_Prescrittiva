# analisi_serie.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Setup percorsi

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset_finale_ETL_QA.xlsx"
OUT_DIR = HERE / "analisi_storiche"
OUT_DIR.mkdir(exist_ok=True)
OUT_GRAFICI = OUT_DIR / "grafici"
OUT_GRAFICI.mkdir(exist_ok=True)
OUT_XLSX = OUT_DIR / "analisi_serie_storiche.xlsx"


# Parametri

MESI_VALIDI = ["GENNAIO","FEBBRAIO","APRILE","MAGGIO","GIUGNO","LUGLIO","AGOSTO"]
MESE_NUM = {m: i+1 for i, m in enumerate(MESI_VALIDI)}


# Caricamento dati

df = pd.read_excel(DATASET)

# Normalizzazioni
df["mese_rif"] = df["mese_rif"].astype(str).str.upper().str.strip()
df = df[df["mese_rif"].isin(MESI_VALIDI)].copy()

# Quantità in uscita
df["outgoing"] = pd.to_numeric(df.get("outgoing", 0), errors="coerce").fillna(0)

# Stock medio (media tra stock e real, se presenti)
for c in ["stock", "real"]:
    if c not in df.columns:
        df[c] = np.nan
df["stock"] = pd.to_numeric(df["stock"], errors="coerce")
df["real"]  = pd.to_numeric(df["real"],  errors="coerce")
df["stock_avg"] = np.nanmean(np.c_[df["stock"].values, df["real"].values], axis=1)
df["stock_avg"] = np.where(np.isnan(df["stock_avg"]), df["stock"], df["stock_avg"])

# Numero mese per ordinamento
df["mese_n"] = df["mese_rif"].map(MESE_NUM)


# Serie mensili per articolo

serie = (df.groupby(["code","mese_rif","mese_n"], as_index=False)
           .agg(consumo_mensile=("outgoing","sum"),
                stock_medio=("stock_avg","mean")))

# Statistiche per articolo
desc = (serie.groupby("code")
        .agg(consumo_medio=("consumo_mensile","mean"),
             consumo_std=("consumo_mensile","std"),
             stock_medio=("stock_medio","mean"))
        .reset_index())

desc["CV_consumo_%"] = (desc["consumo_std"] / desc["consumo_medio"]).replace([np.inf, -np.inf], np.nan) * 100
desc = desc.sort_values("consumo_medio", ascending=False)

# Selezione articoli pilota: top 3 per consumo totale
top_codes = (serie.groupby("code")["consumo_mensile"].sum()
             .sort_values(ascending=False).head(3).index.tolist())

# Esportazione tabelle

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
    serie.sort_values(["code","mese_n"]).to_excel(w, sheet_name="serie_mensili", index=False)
    desc.to_excel(w, sheet_name="statistiche_articoli", index=False)


# Grafici serie temporali (articoli pilota)

for code in top_codes:
    s = serie[serie["code"] == code].sort_values("mese_n")
    plt.figure(figsize=(8, 4.5))
    plt.plot(s["mese_rif"], s["consumo_mensile"], marker="o")
    plt.title(f"Serie storica consumo mensile – Articolo {code}")
    plt.xlabel("Mese (2025)")
    plt.ylabel("Consumo mensile (outgoing)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_GRAFICI / f"serie_{code}.png")
    plt.close()


# Stampa esito essenziale

print("Creati:")
print(" -", OUT_XLSX)
print(" - grafici per:", ", ".join(top_codes))
print("   in:", OUT_GRAFICI)

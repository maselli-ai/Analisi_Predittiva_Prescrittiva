# analisi_variabilita_domanda_leadtime.py

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# Output

OUT_DIR = Path("../output_311")
PLOTS_DIR = OUT_DIR / "grafici_prescrittivi"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# File e colonne (ETL domanda)

FILE_DEMAND  = "../dataset_finale_ETL_QA.xlsx"
COL_ART_DEM  = "code"
COL_MESE_TXT = "mese_rif"
COL_DEMAND   = "outgoing"

# File e colonne (acquisti)

FILE_PURCH   = "C:/Users/Stefania Maselli/Desktop/Analisi_predittiva/acquisti_2025.xlsx"
COL_ART_PUR  = "Cod.Art. No Var."
COL_ORDER    = "Data Doc."
COL_RECEIPT  = "Data Consegna"
COL_LT_DAYS  = "lead_time_days"  # se assente lo calcolo

# Parametri minimi
MIN_MONTHS_DEMAND = 4
MIN_ORDERS_LT     = 3

# Utility

mesi_map = {
    "GENNAIO":1,"FEBBRAIO":2,"MARZO":3,"APRILE":4,"MAGGIO":5,"GIUGNO":6,
    "LUGLIO":7,"AGOSTO":8,"SETTEMBRE":9,"OTTOBRE":10,"NOVEMBRE":11,"DICEMBRE":12
}

def to_numeric_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

# 1) Domanda: sigma_D per articolo

dem = pd.read_excel(FILE_DEMAND, decimal=",")
dem[COL_DEMAND] = to_numeric_series(dem[COL_DEMAND])

dem["Data"] = pd.to_datetime({"year": 2024,
                              "month": dem[COL_MESE_TXT].map(mesi_map),
                              "day": 1})

dem = (
    dem[[COL_ART_DEM, "Data", COL_DEMAND]]
      .dropna(subset=[COL_ART_DEM, "Data"])
      .sort_values([COL_ART_DEM, "Data"])
)

agg_demand = (
    dem.groupby(COL_ART_DEM, dropna=True)
       .agg(
           sigma_D_unita=(COL_DEMAND, lambda x: np.nanstd(x, ddof=1)),
           n_mesi_domanda=(COL_DEMAND, lambda x: x.notna().sum())
       )
       .reset_index()
)
agg_demand.loc[agg_demand["n_mesi_domanda"] < MIN_MONTHS_DEMAND, "sigma_D_unita"] = np.nan


# 2) Acquisti: L e sigma_L per articolo

purch = pd.read_excel(FILE_PURCH)

missing = [c for c in [COL_ART_PUR, COL_ORDER, COL_RECEIPT] if c not in purch.columns]
if missing:
    raise KeyError(f"Mancano colonne in acquisti: {missing}\nPresenti: {list(purch.columns)}")

purch[COL_ORDER]   = pd.to_datetime(purch[COL_ORDER],   dayfirst=True, errors="coerce")
purch[COL_RECEIPT] = pd.to_datetime(purch[COL_RECEIPT], dayfirst=True, errors="coerce")

if COL_LT_DAYS in purch.columns:
    purch["lt_days"] = to_numeric_series(purch[COL_LT_DAYS])
else:
    purch["lt_days"] = (purch[COL_RECEIPT] - purch[COL_ORDER]).dt.days

purch = (
    purch[[COL_ART_PUR, "lt_days"]]
      .dropna()
      .rename(columns={COL_ART_PUR: "code"})
)
purch["code"] = purch["code"].astype(str).str.strip()
purch = purch[purch["lt_days"] >= 0]

agg_lt = (
    purch.groupby("code", dropna=True)
         .agg(
             L_giorni=("lt_days", "mean"),
             sigma_L_giorni=("lt_days", lambda x: np.nanstd(x, ddof=1)),
             n_ordini=("lt_days", "count")
         )
         .reset_index()
)
agg_lt.loc[agg_lt["n_ordini"] < MIN_ORDERS_LT, ["L_giorni", "sigma_L_giorni"]] = np.nan


# 3) Merge ed export

agg_demand = agg_demand.rename(columns={COL_ART_DEM: "code"})
agg_demand["code"] = agg_demand["code"].astype(str).str.strip()

tab = pd.merge(agg_lt, agg_demand, on="code", how="outer")
tab = tab.sort_values(["L_giorni", "sigma_D_unita"], ascending=[False, False])

tab["L_giorni"]       = tab["L_giorni"].round(1)
tab["sigma_L_giorni"] = tab["sigma_L_giorni"].round(1)
tab["sigma_D_unita"]  = tab["sigma_D_unita"].round(2)

fp_tab = OUT_DIR / "Tabella_L_sigmaL_sigmaD.xlsx"
tab.to_excel(fp_tab, index=False)
print("Creato:", fp_tab)


# 4) Grafici lead time complessivi

plt.figure(figsize=(10, 5))
plt.hist(purch["lt_days"].dropna(), bins=20)
plt.title("Distribuzione dei lead time (giorni)")
plt.xlabel("Lead time (giorni)")
plt.ylabel("Frequenza")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "lead_time_hist.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
plt.close()

plt.figure(figsize=(6, 6))
plt.boxplot(purch["lt_days"].dropna(), vert=True)
plt.title("Boxplot lead time (giorni)")
plt.ylabel("Lead time (giorni)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "lead_time_box.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
plt.close()

print("Grafici salvati in:", PLOTS_DIR)


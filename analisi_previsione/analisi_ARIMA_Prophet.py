# analisi_ARIMA_Prophet.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet


# PARAMETRI PRINCIPALI

FILE_XLSX = "../dataset_finale_ETL_QA.xlsx"
COL_ART  = "code"          # codice articolo
COL_CONS = "outgoing"      # consumo o uscita
COL_MESE = "mese_rif"      # mese di riferimento
PERIODI_FUTURI = 3

OUT_DIR = Path("../output_311")
OUT_PLOTS = OUT_DIR / "grafici_arima_prophet"
OUT_PLOTS.mkdir(parents=True, exist_ok=True)


# FUNZIONI DI SUPPORTO

def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    nonzero = y_true != 0
    return (np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]).mean()) * 100

def eval_errors(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE(%)": mape(y_true, y_pred),
    }

def plot_forecast(ts, fcst_index, fcst_values, title, path_png):
    plt.figure(figsize=(12,6))
    plt.plot(ts.index, ts.values, marker="o", label="Domanda reale")
    plt.plot(fcst_index, fcst_values, marker="o", linestyle="--", label="Previsione")
    plt.title(title)
    plt.xlabel("Mese")
    plt.ylabel("Consumo (unità)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_png, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()


# IMPORTAZIONE DATI

df = pd.read_excel(FILE_XLSX)

# Conversione mesi testuali in date (necessaria per ARIMA e Prophet)
mesi_map = {
    "GENNAIO":1,"FEBBRAIO":2,"MARZO":3,"APRILE":4,"MAGGIO":5,"GIUGNO":6,
    "LUGLIO":7,"AGOSTO":8,"SETTEMBRE":9,"OTTOBRE":10,"NOVEMBRE":11,"DICEMBRE":12
}
df["Data"] = pd.to_datetime({"year": 2024, "month": df[COL_MESE].map(mesi_map), "day": 1})


# CICLO SU TUTTI GLI ARTICOLI

articoli = df[COL_ART].unique()
performance = []
previsioni = []

for ARTICOLO in articoli:
    d = df[df[COL_ART] == ARTICOLO].copy()
    d = d.sort_values("Data")
    serie = d.set_index("Data")[COL_CONS].astype(float).asfreq("MS")

    # Esclude articoli con serie troppo corte
    if len(serie.dropna()) <= PERIODI_FUTURI + 2:
        print(f"Articolo {ARTICOLO}: serie troppo corta, saltato.")
        continue

    # Divisione train/test
    train = serie.iloc[:-PERIODI_FUTURI]
    test  = serie.iloc[-PERIODI_FUTURI:]

 
    # Modello ARIMA
    
    order = (1,1,1)
    try:
        model_arima = ARIMA(train, order=order)
        fit_arima = model_arima.fit()
        fcst_arima = fit_arima.forecast(steps=PERIODI_FUTURI)
        err_arima = eval_errors(test, fcst_arima)
    except Exception as e:
        print(f"Errore ARIMA per {ARTICOLO}: {e}")
        continue

    plot_forecast(serie, fcst_arima.index, fcst_arima.values,
                  f"ARIMA{order} – Articolo {ARTICOLO}",
                  OUT_PLOTS / f"{ARTICOLO}_ARIMA.png")


    # Modello PROPHET

    df_p = train.reset_index().rename(columns={"Data": "ds", COL_CONS: "y"})
    try:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(df_p[["ds","y"]])
        future = m.make_future_dataframe(periods=PERIODI_FUTURI, freq="MS")
        forecast_p = m.predict(future).set_index("ds")["yhat"]
        fcst_prophet = forecast_p.iloc[-PERIODI_FUTURI:]
        err_prophet = eval_errors(test, fcst_prophet)
    except Exception as e:
        print(f"Errore Prophet per {ARTICOLO}: {e}")
        continue

    plot_forecast(serie, fcst_prophet.index, fcst_prophet.values,
                  f"Prophet – Articolo {ARTICOLO}",
                  OUT_PLOTS / f"{ARTICOLO}_PROPHET.png")


    # Salvataggio risultati
   
    performance.append({"articolo": ARTICOLO, "modello": f"ARIMA{order}", **err_arima})
    performance.append({"articolo": ARTICOLO, "modello": "Prophet", **err_prophet})

    prev = pd.DataFrame({
        "articolo": ARTICOLO,
        "mese": fcst_arima.index.strftime("%Y-%m"),
        "ARIMA": fcst_arima.values,
        "Prophet": fcst_prophet.values,
        "Reale": test.values
    })
    previsioni.append(prev)


# ESPORTAZIONE RISULTATI

perf_df = pd.DataFrame(performance)
prev_df = pd.concat(previsioni, ignore_index=True)

# Ordina per MAPE crescente (più accurato in alto)
perf_df = perf_df.sort_values(by="MAPE(%)", ascending=True)

perf_df.to_excel(OUT_DIR / "ARIMA_Prophet_performance.xlsx", index=False)
perf_df.to_csv(OUT_DIR / "ARIMA_Prophet_performance.csv", index=False, encoding="utf-8-sig")
prev_df.to_excel(OUT_DIR / "ARIMA_Prophet_previsioni.xlsx", index=False)

print("\nAnalisi completata.")
print(" - Grafici:", OUT_PLOTS)
print(" - Tabella errori:", OUT_DIR / "ARIMA_Prophet_performance.xlsx")
print(" - Previsioni:", OUT_DIR / "ARIMA_Prophet_previsioni.xlsx")

ANALISI PREDITTIVA E PRESCRITTIVA INVENTORY ANALYTICS I

Repository contenente gli script Python, gli output previsionali e prescrittivi e i grafici generati nell’ambito dell’analisi quantitativa dei consumi e dei livelli di stock.

STRUTTURA:

➤ analisi_previsione/
Script Python dedicati ai modelli previsionali:
-  analisi_SMA_WMA.py: Calcolo delle medie mobili semplici e pesate.
-  analisi_ARIMA_Prophet.py: Modelli ARIMA e Prophet per serie storiche.
-  analisi_confronto_modelli.py: Confronto delle prestazioni tra SMA/WMA/ARIMA/Prophet.
-  analisi_variabilita_domanda_leadtime.py: Calcolo della variabilità della domanda e del lead time.
-  calcolo_safety_stock.py: Calcolo della scorta di sicurezza.
-  calcolo_rop.py: Calcolo del Reorder Point.
-  analisi_as_is_to_be.py: Confronto tra stato attuale e modello ottimizzato.


➤output_311/
Output delle analisi previsionali e prescrittive:

Performance e previsioni
- ARIMA_Prophet_performance.xlsx
- SMA_WMA_performance.xlsx
- Confronto_Modelli.xlsx
- ARIMA_Prophet_previsioni.xlsx

Analisi prescrittiva
- Safety_Stock_per_articolo.xlsx
- ROP_per_articolo.xlsx
- AsIs_ToBe_confronto.xlsx
- Tabella_L_sigmaL_sigmaD.xlsx

Grafici
- grafici/: Serie storiche pulite
- grafici_arima_prophet/: Previsioni ARIMA e Prophet per articolo
- grafici_prescrittivi/:  ROP, Safety Stock, distribuzioni e confronti


 ➤ analisi_storiche/
Analisi delle serie storiche e relativi grafici:
- analisi_serie_storiche.xlsx
- Grafici delle serie mensili naturali



➤ File di supporto
- dataset_finale_ETL_QA.xlsx:  Dataset consolidato proveniente dalla pipeline ETL
- acquisti_2025.xlsx: Storico acquisti utilizzato per verifiche aggiuntive




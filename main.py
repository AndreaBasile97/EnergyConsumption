import polars as pl
import pyarrow
import sys
from datetime import datetime
from src.configurator import Configurator
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.Utils import get_args,model_evaluation


#python.exe main.py --pathCSV energy.csv --config MULTI-STEP --target energy_consumption --histFeatures "energy_consumption" --key customer --dateCol date --windowSize 12 --numTargets 12 --spatial None --temporal None
#python.exe main.py --pathCSV "energy_consumption.csv" --config MULTI-STEP --target grid --histFeatures "grid,solar" --key house --dateCol date --windowSize 16 --numTargets 16
# Per SS-DTP in histFeatures avremo "energy_consumption,year,month"

if __name__ == '__main__':
    pathCSV, config, target, histFeatures, key, dateCol, windowSize, numTargets, spatial, temporal = get_args()
    histFeatures = [str(item) for item in histFeatures.split(',')]
    print(pathCSV, config, target, histFeatures, key, dateCol, windowSize, numTargets)


    methods = [RandomForestRegressor(n_jobs=-1, random_state=1),
               KNeighborsRegressor(n_neighbors=3, n_jobs=-1),
               LinearRegression(n_jobs=-1)]


    # istanziamo un oggetto di Classe Configurator con tutte le informazioni necessarie alla trasformazione del datset
    conf = Configurator(configuration=config, windows_size=windowSize, n_targets=numTargets,target=target,key=key,
                        dateCol=dateCol, histFeatures=histFeatures, spatial=spatial, temporal=temporal)

    dataset = pl.read_csv(pathCSV, try_parse_dates=True, use_pyarrow=True)

    dataset = dataset.drop("city")

    # aggiungiamo le feature temporali di anno e mese
    dataset = dataset.with_columns(pl.col("date").dt.year().alias("year"))
    dataset = dataset.with_columns(pl.col("date").dt.month().alias("month"))

    # trasformazione del dataset nel setting specificato
    df, config = conf.transform(dataset)

    # settiamo il range di date che verranno utilizzate per la creazione del test set
    start_pred_date = datetime.strptime('2019-05-20 00:00:00', '%Y-%m-%d %H:%M:%S')
    end_pred_date = datetime.strptime('2019-05-21 20:00:00', '%Y-%m-%d %H:%M:%S')

    # valutiamo ogni metodo di regressione sul test set e memorizziamo l'output su file
    for method in methods:
        orig, pred, dates, key_val = conf.prediction(df, start_pred_date, end_pred_date,method)
        score = model_evaluation(orig, pred, dates, config, method, start_pred_date,end_pred_date, key_val, numTargets, target, key)




